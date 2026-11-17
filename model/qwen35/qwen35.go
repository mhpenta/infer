// Package qwen35 runs Qwen3.5's text decoder from Hugging Face weights, on
// the CPU, with the sequence-classification head its classifier checkpoints
// add (Qwen3_5ForSequenceClassification), such as OpenJev's NLI
// cross-encoders.
//
// Qwen3.5 mixes two kinds of layers. Most are linear attention, Gated
// DeltaNet: a depthwise causal convolution over the projected queries, keys
// and values, then the gated delta rule (ops.GatedDeltaRule), whose state is
// a fixed-size matrix per head instead of a cache that grows with the
// sequence. Every full_attention_interval-th layer is grouped-query
// attention with a sigmoid output gate, per-head query and key RMSNorm, and
// rotary embeddings on the first quarter of each head. Its RMSNorm weights
// are stored zero-centered (the scale is 1+w), and the feed-forward is
// SwiGLU, as in Llama.
//
// Every product runs in float32. Weights stored as bf16 stay bf16 in memory,
// packed with ops.PackBF16, or are quantized to int8 (Options.Int8). The
// vision tower that Qwen3.5 checkpoints carry is not loaded, and the
// language-model head is not either: the package classifies and returns
// hidden states, it does not generate.
package qwen35

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// Config is the subset of the text configuration the model uses: config.json's
// text_config, or config.json itself for a text-only checkpoint.
type Config struct {
	ModelType         string   `json:"model_type"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	LayerTypes        []string `json:"layer_types"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	VocabSize         int      `json:"vocab_size"`
	RMSNormEps        float64  `json:"rms_norm_eps"`
	HiddenAct         string   `json:"hidden_act"`
	AttentionBias     bool     `json:"attention_bias"`
	AttnOutputGate    *bool    `json:"attn_output_gate"`
	PadTokenID        *int     `json:"pad_token_id"`

	LinearConvKernelDim int `json:"linear_conv_kernel_dim"`
	LinearKeyHeadDim    int `json:"linear_key_head_dim"`
	LinearValueHeadDim  int `json:"linear_value_head_dim"`
	LinearNumKeyHeads   int `json:"linear_num_key_heads"`
	LinearNumValueHeads int `json:"linear_num_value_heads"`

	RopeParameters struct {
		RopeType            string  `json:"rope_type"`
		RopeTheta           float64 `json:"rope_theta"`
		PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	} `json:"rope_parameters"`

	// Labels are the classifier's classes, in the score head's order
	// (config.json's id2label), or nil for a model without one.
	Labels []string `json:"-"`
}

// Layer types, from layer_types.
const (
	LinearAttention = "linear_attention"
	FullAttention   = "full_attention"
)

func (c Config) qDim() int     { return c.NumAttentionHeads * c.HeadDim }
func (c Config) kvDim() int    { return c.NumKeyValueHeads * c.HeadDim }
func (c Config) keyDim() int   { return c.LinearNumKeyHeads * c.LinearKeyHeadDim }
func (c Config) valueDim() int { return c.LinearNumValueHeads * c.LinearValueHeadDim }

// convDim is the width of the linear layers' convolution: queries, keys and
// values.
func (c Config) convDim() int { return 2*c.keyDim() + c.valueDim() }

// rotaryDim is how many leading dimensions of each attention head rotate.
func (c Config) rotaryDim() int {
	return int(float64(c.HeadDim) * c.RopeParameters.PartialRotaryFactor)
}

func (c Config) deltaArgs() ops.DeltaRuleArgs {
	return ops.DeltaRuleArgs{
		KeyHeads: c.LinearNumKeyHeads, ValueHeads: c.LinearNumValueHeads,
		KeyDim: c.LinearKeyHeadDim, ValueDim: c.LinearValueHeadDim,
	}
}

// validate rejects configurations this package does not implement rather
// than running them wrong.
func (c Config) validate() error {
	switch {
	case c.ModelType != "qwen3_5_text" && c.ModelType != "qwen3_5":
		return fmt.Errorf("model_type %q is not qwen3_5_text", c.ModelType)
	case c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.VocabSize <= 0:
		return errors.New("config is missing sizes")
	case len(c.LayerTypes) != c.NumHiddenLayers:
		return fmt.Errorf("%d layer types for %d layers", len(c.LayerTypes), c.NumHiddenLayers)
	case c.HiddenAct != "silu":
		return fmt.Errorf("hidden_act %q is not supported", c.HiddenAct)
	case c.AttentionBias:
		return errors.New("attention biases are not supported")
	case c.AttnOutputGate != nil && !*c.AttnOutputGate:
		return errors.New("attention without an output gate is not supported")
	case c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0:
		return fmt.Errorf("%d attention heads do not split into %d key/value heads", c.NumAttentionHeads, c.NumKeyValueHeads)
	case c.RopeParameters.RopeType != "default":
		return fmt.Errorf("rope_type %q is not supported", c.RopeParameters.RopeType)
	case c.rotaryDim() <= 0 || c.rotaryDim()%2 != 0 || c.rotaryDim() > c.HeadDim:
		return fmt.Errorf("partial_rotary_factor %g of head_dim %d is not an even rotary width", c.RopeParameters.PartialRotaryFactor, c.HeadDim)
	case c.LinearNumKeyHeads <= 0 || c.LinearNumValueHeads%c.LinearNumKeyHeads != 0 || c.LinearKeyHeadDim <= 0 || c.LinearValueHeadDim <= 0:
		return fmt.Errorf("%d linear value heads do not share %d key heads", c.LinearNumValueHeads, c.LinearNumKeyHeads)
	case c.LinearConvKernelDim < 1:
		return fmt.Errorf("linear_conv_kernel_dim %d", c.LinearConvKernelDim)
	}
	for i, t := range c.LayerTypes {
		if t != LinearAttention && t != FullAttention {
			return fmt.Errorf("layer %d has type %q", i, t)
		}
	}
	return nil
}

// readConfig reads config.json from dir: its text_config, if it has one, and
// the classifier's labels.
func readConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Config{}, err
	}
	var top struct {
		TextConfig json.RawMessage   `json:"text_config"`
		ID2Label   map[string]string `json:"id2label"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return Config{}, fmt.Errorf("parse config.json: %w", err)
	}
	text := raw
	if len(top.TextConfig) > 0 {
		text = top.TextConfig
	}
	var c Config
	if err := json.Unmarshal(text, &c); err != nil {
		return Config{}, fmt.Errorf("parse config.json: %w", err)
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = 1e-6
	}
	if c.HiddenAct == "" {
		c.HiddenAct = "silu"
	}
	if c.RopeParameters.RopeType == "" {
		c.RopeParameters.RopeType = "default"
	}
	if c.RopeParameters.PartialRotaryFactor == 0 {
		c.RopeParameters.PartialRotaryFactor = 1
	}
	if len(top.ID2Label) > 0 {
		c.Labels = make([]string, len(top.ID2Label))
		for id, label := range top.ID2Label {
			i, err := strconv.Atoi(id)
			if err != nil || i < 0 || i >= len(c.Labels) || c.Labels[i] != "" {
				return Config{}, fmt.Errorf("id2label has id %q of %d labels", id, len(top.ID2Label))
			}
			c.Labels[i] = label
		}
	}
	return c, nil
}

// Model is a loaded Qwen3.5 text decoder. It is safe for concurrent use: a
// forward pass keeps its state per call.
type Model struct {
	Config Config

	files  []*safetensors.File
	embed  []uint16       // bf16 token embeddings [vocab, hidden], or nil with embedF32
	embedF *tensor.Tensor // float32 token embeddings, when not stored as bf16
	layers []layer
	norm   *tensor.Tensor // final RMSNorm scale (1+w)
	score  *ops.Packed    // classifier head [labels, hidden], or nil
	rope   *ops.RoPE      // over the rotary dimensions only
}

type layer struct {
	full             bool
	attnNorm, ffNorm *tensor.Tensor // RMSNorm scales (1+w)

	// Full attention: q (all heads), the output gate (all heads), k and v,
	// stacked; the q_proj rows, which alternate one head's queries and gate,
	// are reordered at load.
	qgkv         *ops.Packed
	qNorm, kNorm *tensor.Tensor // per-head RMSNorm scales (1+w)
	out          *ops.Packed

	// Linear attention: the convolution's input (queries, keys, values),
	// then z, b and a, stacked.
	in        *ops.Packed
	conv      []float32 // depthwise convolution weights [convDim, kernel]
	negA      []float32 // −exp(A_log) per value head
	dtBias    []float32 // per value head
	headNorm  *tensor.Tensor
	linearOut *ops.Packed

	gateUp *ops.Packed // gate and up projections stacked
	down   *ops.Packed
}

// Options control how LoadWith prepares a model.
type Options struct {
	// Int8 quantizes the linear layers to int8 with one scale per output
	// (ops.PackInt8) instead of keeping the checkpoint's precision. The
	// token embeddings, the convolution and the classifier head keep theirs.
	Int8 bool
}

// Load reads config.json and the weights (model.safetensors, or the shards
// model.safetensors.index.json lists) from dir.
func Load(dir string) (*Model, error) { return LoadWith(dir, Options{}) }

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (*Model, error) {
	config, err := readConfig(dir)
	if err != nil {
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	files, byName, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	m, err := newModel(config, byName, opts)
	if err != nil {
		for _, f := range files {
			f.Close()
		}
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	m.files = files
	return m, nil
}

// prefix returns the language model's tensor-name prefix: the classifier
// and multimodal checkpoints nest it under model.language_model.
func prefix(byName map[string]*safetensors.File) string {
	if _, ok := byName["model.language_model.embed_tokens.weight"]; ok {
		return "model.language_model."
	}
	return "model."
}

func newModel(c Config, byName map[string]*safetensors.File, opts Options) (*Model, error) {
	l := loader{files: byName, int8: opts.Int8}
	m := &Model{Config: c, rope: ops.NewRoPE(c.rotaryDim(), c.RopeParameters.RopeTheta)}
	h, ff := c.HiddenSize, c.IntermediateSize
	base := prefix(byName)
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("%slayers.%d.", base, i)
		ly := layer{
			full:     c.LayerTypes[i] == FullAttention,
			attnNorm: l.scale(p+"input_layernorm.weight", h),
			ffNorm:   l.scale(p+"post_attention_layernorm.weight", h),
			gateUp:   l.stacked(h, []string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, []int{ff, ff}),
			down:     l.linear(p+"mlp.down_proj.weight", h, ff),
		}
		if ly.full {
			ly.qgkv = l.attentionIn(c, p+"self_attn.")
			ly.qNorm = l.scale(p+"self_attn.q_norm.weight", c.HeadDim)
			ly.kNorm = l.scale(p+"self_attn.k_norm.weight", c.HeadDim)
			ly.out = l.linear(p+"self_attn.o_proj.weight", h, c.qDim())
		} else {
			a, nv := p+"linear_attn.", c.LinearNumValueHeads
			ly.in = l.stacked(h, []string{a + "in_proj_qkv.weight", a + "in_proj_z.weight", a + "in_proj_b.weight", a + "in_proj_a.weight"},
				[]int{c.convDim(), c.valueDim(), nv, nv})
			if conv := l.tensor(a+"conv1d.weight", c.convDim(), 1, c.LinearConvKernelDim); conv != nil {
				ly.conv = conv.Data
			}
			if aLog := l.tensor(a+"A_log", nv); aLog != nil {
				ly.negA = make([]float32, nv)
				for j, v := range aLog.Data {
					ly.negA[j] = -float32(math.Exp(float64(v))) // A_log.float().exp(), then negated
				}
			}
			if dt := l.tensor(a+"dt_bias", nv); dt != nil {
				ly.dtBias = dt.Data
			}
			ly.headNorm = l.tensor(a+"norm.weight", c.LinearValueHeadDim)
			ly.linearOut = l.linear(a+"out_proj.weight", h, c.valueDim())
		}
		m.layers = append(m.layers, ly)
	}
	m.norm = l.scale(base+"norm.weight", h)
	embedName := base + "embed_tokens.weight"
	embed := l.tensor(embedName, c.VocabSize, h)
	if c.Labels != nil {
		m.score = l.linear("score.weight", len(c.Labels), h)
	}
	if l.err != nil {
		return nil, l.err
	}
	m.embedF = embed
	if l.dtype(embedName) == "BF16" {
		m.embed, m.embedF = make([]uint16, len(embed.Data)), nil
		for i, v := range embed.Data {
			m.embed[i] = uint16(math.Float32bits(v) >> 16) // exact: the value came from bf16
		}
	}
	return m, nil
}

// embedRow writes token id's embedding to dst.
func (m *Model) embedRow(id int, dst []float32) {
	if m.embed == nil {
		copy(dst, m.embedF.Row(id))
		return
	}
	h := m.Config.HiddenSize
	for j, b := range m.embed[id*h : (id+1)*h] {
		dst[j] = math.Float32frombits(uint32(b) << 16)
	}
}

// Close releases the weight files.
func (m *Model) Close() error {
	var errs []error
	for _, f := range m.files {
		errs = append(errs, f.Close())
	}
	return errors.Join(errs...)
}

func (m *Model) checkIDs(ids []int) error {
	if len(ids) == 0 {
		return errors.New("qwen35: no input ids")
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.VocabSize {
			return fmt.Errorf("qwen35: id %d outside the vocabulary of %d", id, m.Config.VocabSize)
		}
	}
	return nil
}

// HiddenStates runs ids and returns num_hidden_layers+1 states [len(ids),
// hidden]: the input to each layer (the first is the embeddings), then the
// final normalized output, as transformers' output_hidden_states.
func (m *Model) HiddenStates(ids []int) ([]*tensor.Tensor, error) {
	if err := m.checkIDs(ids); err != nil {
		return nil, err
	}
	var states []*tensor.Tensor
	m.newState(len(ids)).forward(ids, func(x *tensor.Tensor) {
		states = append(states, tensor.FromData(slices.Clone(x.Data), x.Shape...))
	})
	return states, nil
}

// Classify runs ids and returns the score head's logits at the last token,
// one per Config.Labels, as Qwen3_5ForSequenceClassification pools an
// unpadded sequence. It fails for a model without a classifier head.
func (m *Model) Classify(ids []int) ([]float32, error) {
	if m.score == nil {
		return nil, errors.New("qwen35: the checkpoint has no classifier head (score.weight)")
	}
	if err := m.checkIDs(ids); err != nil {
		return nil, err
	}
	y := m.newState(len(ids)).forward(ids, nil)
	return m.scoreRow(y.Row(y.Rows() - 1)), nil
}

// ClassifyShared is Classify on prefix followed by each of suffixes, running
// prefix once: each suffix continues from the state after prefix. Results
// are bit-identical to Classify on the joined ids, since every operation
// computes a row the same way whichever rows run with it, and the linear
// layers' recurrence runs positions in order either way. Every suffix must
// be non-empty; prefix may be empty.
func (m *Model) ClassifyShared(prefix []int, suffixes [][]int) ([][]float32, error) {
	if m.score == nil {
		return nil, errors.New("qwen35: the checkpoint has no classifier head (score.weight)")
	}
	longest := 0
	for _, s := range suffixes {
		if err := m.checkIDs(s); err != nil {
			return nil, err
		}
		longest = max(longest, len(s))
	}
	st := m.newState(len(prefix) + longest)
	if len(prefix) > 0 {
		if err := m.checkIDs(prefix); err != nil {
			return nil, err
		}
		st.forward(prefix, nil)
	}
	saved := st.save()
	out := make([][]float32, len(suffixes))
	for i, s := range suffixes {
		st.restore(saved)
		y := st.forward(s, nil)
		out[i] = m.scoreRow(y.Row(y.Rows() - 1))
	}
	return out, nil
}

// scoreRow applies the classifier head to one final hidden state.
func (m *Model) scoreRow(y []float32) []float32 {
	out := tensor.New(1, m.score.N())
	ops.LinearInto(out, tensor.FromData(y, 1, len(y)), m.score, nil)
	return out.Data
}
