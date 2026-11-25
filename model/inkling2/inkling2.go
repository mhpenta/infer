// Package inkling2 runs Thinking Machines' Inkling 2 Mini text decoder
// (Inkling2LoopForConditionalGeneration's language model) on the CPU, from
// the checkpoint as it ships: experts in NVFP4 and MXFP8, decoded as they
// are used, everything else bf16 or float32.
//
// The model is a sparse mixture-of-experts decoder whose middle layers loop:
// layers [core_start, core_end) run num_passes times with shared weights,
// every pass restarting from the output of the layers before the core and,
// after the first, reading messages from the previous pass. Its pieces,
// each as transformers' modeling_inkling2_loop.py computes it:
//
//   - Attention residuals: a layer's input is not the running sum but a
//     softmax mix, with a learned query, of the block states (the inputs of
//     every n_layers_per_attn_res_block-th layer) and the current partial
//     sum, each RMS-normalized. Full-attention layers mix three inputs, for
//     their queries, keys and values.
//   - Attention: grouped-query, sliding-window in most layers and full in
//     the rest, with no rotary embedding. Positions enter through a learned
//     relative bias (each head's relative states, from the layer input,
//     weigh a bank of bias-versus-distance profiles), a per-token sink
//     logit, short causal convolutions over the keys and values (with
//     per-token taps in the full layers), L2-clipped queries and keys,
//     RMS-normalized values, and a sigmoid output gate.
//   - Short convolutions after attention and after the experts.
//   - Experts: a sigmoid router picks 8 of 128 by score plus bias; each is a
//     SwiGLU whose gate and up projections are softcapped.
//   - Message gates: in passes after the first, each core layer mixes into
//     its query, key, value and expert inputs a message, read by attention
//     over the previous pass's last block state and output.
//
// Products run in float32. The vision tower and the multi-token prediction
// head are not loaded.
package inkling2

import (
	"errors"
	"fmt"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/sample"
	"github.com/mhpenta/infer/tensor"
)

// Model is a loaded Inkling 2 text decoder. It is safe for concurrent use:
// each call keeps its own state.
type Model struct {
	Config Config

	files          []*safetensors.File
	embed          []byte    // bf16 token embeddings [vocab, hidden], a view into the mapping
	embedNorm      []float32 // RMSNorm weight applied to every embedding
	layers         []layer
	messageQueries [][]float32 // per core layer: query, key, value and expert message queries [4·hidden]
	finalQuery     []float32   // the attention residual before the final norm
	norm           []float32
	unembed        *ops.Packed // [unpadded vocab, hidden]
}

type layer struct {
	index            int
	sliding, loop    bool
	heads, kvHeads   int
	headDim          int
	relExtent        int
	attnQueries      []float32 // the attention input's residual queries: 1 (sliding) or 3 (full) × hidden; nil for layer 0
	mlpQuery         []float32
	inNorm           []float32 // sliding layers: RMSNorm of the attention input
	qNorm            []float32 // full layers: RMSNorm of the query, key and value inputs
	kNorm, vNorm     []float32
	mlpNorm          []float32
	q                *ops.Packed // queries, output gate, relative states, sinks, stacked
	k, v             *ops.Packed // keys and values, and in full layers their convolutions' per-token taps
	qScale, relScale []float32   // scales before the L2 clips [heads·headDim], [heads·d_rel]
	relProj          []float32   // the bias profiles [d_rel, relExtent]
	kConv, vConv     []float32   // sliding: [kv, kernel]; full: static taps [kernel, kv]
	out              *ops.Packed
	postAttnConv     []float32 // [hidden, kernel]
	postMLPConv      []float32
	router           *ops.Packed
	routerBias       []float32
	gateUp, down     []*ops.Microscaled // per expert
	gates            [4]messageGate     // core layers: query, key, value, expert inputs
}

// messageGate mixes a message into a hidden state: (1−f)·h + g·m, with f and
// g tanh of rank-r projections of the normalized pair.
type messageGate struct {
	down         *ops.Packed // [rank, 2·hidden]
	forget, gain *ops.Packed // [hidden, rank]
}

// Load reads config.json and the weights from dir.
func Load(dir string) (*Model, error) {
	config, err := readConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("inkling2: %w", err)
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("inkling2: %w", err)
	}
	files, byName, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("inkling2: %w", err)
	}
	m, err := newModel(config, byName)
	if err != nil {
		for _, f := range files {
			f.Close()
		}
		return nil, fmt.Errorf("inkling2: %w", err)
	}
	m.files = files
	return m, nil
}

// Close releases the weight files. The model must not be used after.
func (m *Model) Close() error {
	var errs []error
	for _, f := range m.files {
		errs = append(errs, f.Close())
	}
	return errors.Join(errs...)
}

func (m *Model) checkIDs(ids []int) error {
	if len(ids) == 0 {
		return errors.New("inkling2: no input ids")
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.VocabSize {
			return fmt.Errorf("inkling2: id %d outside the vocabulary of %d", id, m.Config.VocabSize)
		}
	}
	return nil
}

// HiddenStates runs ids and returns Config.Visits()+1 states [len(ids),
// hidden]: the input to every layer in the order the layers run (the first
// is the normalized embeddings; the core's layers appear once per pass),
// then the final normalized output.
func (m *Model) HiddenStates(ids []int) ([]*tensor.Tensor, error) {
	if err := m.checkIDs(ids); err != nil {
		return nil, err
	}
	var states []*tensor.Tensor
	m.newState().forward(ids, func(x *tensor.Tensor) {
		states = append(states, tensor.FromData(append([]float32(nil), x.Data...), x.Shape...))
	})
	return states, nil
}

// Logits runs ids and returns the logits at every position [len(ids),
// unpadded vocab], softcapped as the model's head does.
func (m *Model) Logits(ids []int) (*tensor.Tensor, error) {
	if err := m.checkIDs(ids); err != nil {
		return nil, err
	}
	y := m.newState().forward(ids, nil)
	out := tensor.New(y.Rows(), m.Config.UnpaddedVocabSize)
	m.logitsInto(out, y)
	return out, nil
}

// logitsInto sets out [n, unpadded vocab] to the softcapped logits of the
// final states y [n, hidden].
func (m *Model) logitsInto(out, y *tensor.Tensor) {
	ops.LinearInto(out, y, m.unembed, nil)
	softcapInPlace(out.Data, float32(m.Config.FinalLogitSoftcapping))
}

// Generate decodes greedily after ids (an already rendered and tokenized
// prompt) and returns the new tokens: up to maxNew, ending with the end
// token (Config.EOS) if one is produced.
func (m *Model) Generate(ids []int, maxNew int) ([]int, error) {
	var out []int
	err := m.GenerateWith(ids, maxNew, nil, func(token int, _ []float32) bool {
		out = append(out, token)
		return true
	})
	return out, err
}

// GenerateWith is Generate choosing each token with s instead of greedily (a
// nil s is greedy), calling fn with each new token and the logits it was
// chosen from as soon as it is chosen; fn returning false stops.
func (m *Model) GenerateWith(ids []int, maxNew int, s *sample.Sampler, fn func(token int, logits []float32) bool) error {
	if err := m.checkIDs(ids); err != nil {
		return err
	}
	if maxNew < 0 {
		return fmt.Errorf("inkling2: maxNew %d", maxNew)
	}
	if maxNew == 0 {
		return nil
	}
	st := m.newState()
	seq := append([]int(nil), ids...)
	logits := tensor.New(1, m.Config.UnpaddedVocabSize)
	next := ids
	for range maxNew {
		y := st.forward(next, nil)
		m.logitsInto(logits, tensor.FromData(y.Row(y.Rows()-1), 1, y.Cols()))
		token := ops.Argmax(logits.Data)
		if s != nil {
			var err error
			if token, err = s.Next(logits.Data, seq); err != nil {
				return fmt.Errorf("inkling2: %w", err)
			}
		}
		seq = append(seq, token)
		if !fn(token, logits.Data) || token == m.Config.EOS {
			return nil
		}
		next = []int{token}
	}
	return nil
}
