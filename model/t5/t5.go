// Package t5 implements the T5 encoder-decoder for inference: encoding, the
// decoder, and greedy generation.
//
// The architecture, as in Hugging Face's T5ForConditionalGeneration:
//
//   - Pre-norm residual blocks with RMSNorm (no mean, no bias) and no biases
//     in any linear layer.
//   - Attention without the 1/√d scaling, plus a learned relative position
//     bias: offsets between positions are bucketed (exactly for small
//     offsets, logarithmically up to relative_attention_max_distance) and
//     each bucket has a bias per head. Only the first block of each stack
//     owns the table; every block reuses its bias. The encoder's buckets are
//     bidirectional, the decoder's look only backwards.
//   - Decoder blocks add cross-attention over the encoder output, without a
//     position bias.
//   - The output layer is the embedding matrix (tied), applied after scaling
//     the decoder output by d_model^-0.5.
//
// Weights use the Hugging Face tensor names.
//
// On Apple silicon the model can run on the GPU instead (Options.Device,
// metal.go): by default it does when the GPU is available and suits the
// model.
package t5

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// Config is the subset of a Hugging Face T5 config.json the model needs.
type Config struct {
	VocabSize                    int     `json:"vocab_size"`
	DModel                       int     `json:"d_model"`
	DKV                          int     `json:"d_kv"`
	DFF                          int     `json:"d_ff"`
	NumLayers                    int     `json:"num_layers"`
	NumDecoderLayers             int     `json:"num_decoder_layers"`
	NumHeads                     int     `json:"num_heads"`
	RelativeAttentionNumBuckets  int     `json:"relative_attention_num_buckets"`
	RelativeAttentionMaxDistance int     `json:"relative_attention_max_distance"`
	LayerNormEpsilon             float64 `json:"layer_norm_epsilon"`
	FeedForwardProj              string  `json:"feed_forward_proj"`
	TieWordEmbeddings            *bool   `json:"tie_word_embeddings"`
	DecoderStartTokenID          int     `json:"decoder_start_token_id"`
	EOSTokenID                   int     `json:"eos_token_id"`
}

// inner is the width of the attention projections: heads × d_kv.
func (c Config) inner() int { return c.NumHeads * c.DKV }

func (c Config) tied() bool { return c.TieWordEmbeddings == nil || *c.TieWordEmbeddings }

// withDefaults fills in the values transformers assumes when config.json
// leaves them out.
func (c Config) withDefaults() Config {
	if c.NumDecoderLayers == 0 {
		c.NumDecoderLayers = c.NumLayers
	}
	if c.RelativeAttentionMaxDistance == 0 {
		c.RelativeAttentionMaxDistance = 128
	}
	if c.FeedForwardProj == "" {
		c.FeedForwardProj = "relu"
	}
	if c.LayerNormEpsilon == 0 {
		c.LayerNormEpsilon = 1e-6
	}
	return c
}

func (c Config) validate() error {
	switch {
	case c.DModel <= 0 || c.NumHeads <= 0 || c.DKV <= 0 || c.DFF <= 0 || c.NumLayers <= 0:
		return errors.New("d_model, num_heads, d_kv, d_ff and num_layers must be positive")
	case c.RelativeAttentionNumBuckets < 4:
		return fmt.Errorf("relative_attention_num_buckets %d is too small", c.RelativeAttentionNumBuckets)
	case c.FeedForwardProj != "relu":
		return fmt.Errorf("feed_forward_proj %q is not supported (want relu)", c.FeedForwardProj)
	}
	return nil
}

// Model is a loaded T5 model. It is safe for concurrent use.
type Model struct {
	Config Config

	shared  *tensor.Tensor // token embeddings [vocab, d_model]
	lmHead  *ops.Packed    // output layer: the embeddings when tied
	encoder []encoderLayer
	encNorm *tensor.Tensor
	encBias *tensor.Tensor // relative position bias table [buckets, heads]
	decoder []decoderLayer
	decNorm *tensor.Tensor
	decBias *tensor.Tensor
	weights *safetensors.File

	gpu    *gpuModel // the weights on the GPU, when the model runs there
	device string
}

type encoderLayer struct {
	attnNorm *tensor.Tensor
	qkv, out *ops.Packed // q, k and v stacked; output projection
	ffNorm   *tensor.Tensor
	wi, wo   *ops.Packed
}

type decoderLayer struct {
	selfNorm  *tensor.Tensor
	selfQKV   *ops.Packed
	selfOut   *ops.Packed
	crossNorm *tensor.Tensor
	crossQ    *ops.Packed
	crossKV   *ops.Packed // k and v stacked, applied to the encoder output
	crossOut  *ops.Packed
	ffNorm    *tensor.Tensor
	wi, wo    *ops.Packed
}

// Options control how LoadWith prepares a model.
type Options struct {
	// Device is where the model runs: accel.CPU, accel.Metal (an Apple
	// silicon GPU), or accel.Auto. Empty means $INFER_DEVICE, or Auto when
	// that is unset too. Auto runs on Metal when the GPU is available,
	// passes its self-check, and suits the model, and on the CPU otherwise;
	// asking for Metal when it doesn't suit is an error.
	Device string
}

// Load reads config.json and model.safetensors from a model directory.
func Load(dir string) (*Model, error) {
	return LoadWith(dir, Options{})
}

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (*Model, error) {
	device, err := accel.Resolve(opts.Device)
	if err != nil {
		return nil, fmt.Errorf("t5: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("t5: %w", err)
	}
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("t5: parse config.json: %w", err)
	}
	config = config.withDefaults()
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("t5: %w", err)
	}
	weights, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("t5: %w", err)
	}
	var m *Model
	if device != accel.CPU {
		var g *gpuModel
		if g, err = openGPU(config, weights); err == nil {
			l := loader{weights: weights}
			m = &Model{
				Config:  config,
				encBias: l.tensor("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", config.RelativeAttentionNumBuckets, config.NumHeads),
				decBias: l.tensor("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", config.RelativeAttentionNumBuckets, config.NumHeads),
				weights: weights,
				gpu:     g,
				device:  fmt.Sprintf("%s (%s)", accel.Metal, g.gpu.Device.Name()),
			}
			if err = l.err; err != nil {
				g.release()
				m = nil
			}
		} else if device == accel.Metal {
			err = fmt.Errorf("cannot run on Metal: %w", err)
		} else {
			err = nil // Auto: the CPU, then
		}
	}
	if m == nil && err == nil {
		m, err = newModel(config, weights)
	}
	if err != nil {
		weights.Close()
		return nil, fmt.Errorf("t5: %w", err)
	}
	return m, nil
}

func newModel(c Config, weights *safetensors.File) (*Model, error) {
	l := loader{weights: weights, half: ops.HalfPanels()}
	d, inner := c.DModel, c.inner()
	m := &Model{
		Config:  c,
		shared:  l.tensor("shared.weight", c.VocabSize, d),
		encNorm: l.tensor("encoder.final_layer_norm.weight", d),
		encBias: l.tensor("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", c.RelativeAttentionNumBuckets, c.NumHeads),
		decNorm: l.tensor("decoder.final_layer_norm.weight", d),
		decBias: l.tensor("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", c.RelativeAttentionNumBuckets, c.NumHeads),
		weights: weights,
		device:  accel.CPU,
	}
	if c.tied() {
		m.lmHead = l.pack(m.shared, "shared.weight")
	} else {
		m.lmHead = l.linear("lm_head.weight", c.VocabSize, d)
	}
	for i := range c.NumLayers {
		p := fmt.Sprintf("encoder.block.%d.layer.", i)
		m.encoder = append(m.encoder, encoderLayer{
			attnNorm: l.tensor(p+"0.layer_norm.weight", d),
			qkv:      l.stacked(inner, d, p+"0.SelfAttention.q.weight", p+"0.SelfAttention.k.weight", p+"0.SelfAttention.v.weight"),
			out:      l.linear(p+"0.SelfAttention.o.weight", d, inner),
			ffNorm:   l.tensor(p+"1.layer_norm.weight", d),
			wi:       l.linear(p+"1.DenseReluDense.wi.weight", c.DFF, d),
			wo:       l.linear(p+"1.DenseReluDense.wo.weight", d, c.DFF),
		})
	}
	for i := range c.NumDecoderLayers {
		p := fmt.Sprintf("decoder.block.%d.layer.", i)
		m.decoder = append(m.decoder, decoderLayer{
			selfNorm:  l.tensor(p+"0.layer_norm.weight", d),
			selfQKV:   l.stacked(inner, d, p+"0.SelfAttention.q.weight", p+"0.SelfAttention.k.weight", p+"0.SelfAttention.v.weight"),
			selfOut:   l.linear(p+"0.SelfAttention.o.weight", d, inner),
			crossNorm: l.tensor(p+"1.layer_norm.weight", d),
			crossQ:    l.linear(p+"1.EncDecAttention.q.weight", inner, d),
			crossKV:   l.stacked(inner, d, p+"1.EncDecAttention.k.weight", p+"1.EncDecAttention.v.weight"),
			crossOut:  l.linear(p+"1.EncDecAttention.o.weight", d, inner),
			ffNorm:    l.tensor(p+"2.layer_norm.weight", d),
			wi:        l.linear(p+"2.DenseReluDense.wi.weight", c.DFF, d),
			wo:        l.linear(p+"2.DenseReluDense.wo.weight", d, c.DFF),
		})
	}
	if l.err != nil {
		return nil, l.err
	}
	return m, nil
}

// Device returns where the model runs: "cpu", or "metal (" and the GPU's
// name ")".
func (m *Model) Device() string { return m.device }

// Close releases the memory-mapped weights and any GPU memory.
func (m *Model) Close() error {
	if m.gpu != nil {
		m.gpu.release()
	}
	return m.weights.Close()
}

func (m *Model) checkIDs(ids []int) error {
	if len(ids) == 0 {
		return errors.New("t5: empty input")
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.VocabSize {
			return fmt.Errorf("t5: token id %d outside vocab of %d", id, m.Config.VocabSize)
		}
	}
	return nil
}

// Encode runs the encoder on input ids and returns its final, normalized
// output [len(ids), d_model].
func (m *Model) Encode(ids []int) (*tensor.Tensor, error) {
	states, err := m.encode(ids, false)
	if err != nil {
		return nil, err
	}
	return states[len(states)-1], nil
}

// EncoderHiddenStates is Encode returning every hidden state, as
// transformers' encoder_hidden_states: the input to each layer (the first is
// the embedding output), then the final, normalized output.
func (m *Model) EncoderHiddenStates(ids []int) ([]*tensor.Tensor, error) {
	return m.encode(ids, true)
}

func (m *Model) encode(ids []int, keepAll bool) ([]*tensor.Tensor, error) {
	if err := m.checkIDs(ids); err != nil {
		return nil, err
	}
	if m.gpu != nil {
		return m.gpu.encode(m, ids, keepAll)
	}
	c := m.Config
	seq, inner := len(ids), c.inner()
	x := ops.Gather(m.shared, ids)
	bias := positionBias(m.encBias, c, seq, seq, 0, true)
	var states []*tensor.Tensor
	h := tensor.New(seq, c.DModel)
	qkv, ctx, y := tensor.New(seq, 3*inner), tensor.New(seq, inner), tensor.New(seq, c.DModel)
	ff := tensor.New(seq, c.DFF)
	for _, l := range m.encoder {
		if keepAll {
			states = append(states, clone(x))
		}
		m.norm(h, x, l.attnNorm)
		ops.LinearInto(qkv, h, l.qkv, nil)
		ops.AttentionStrided(ctx.Data, inner, qkv.Data, 3*inner, qkv.Data[inner:], 3*inner, qkv.Data[2*inner:], 3*inner,
			seq, seq, inner, ops.AttentionArgs{Heads: c.NumHeads, Scale: 1, Bias: bias})
		ops.LinearInto(y, ctx, l.out, nil)
		ops.Add(x.Data, y.Data)

		m.norm(h, x, l.ffNorm)
		ops.LinearInto(ff, h, l.wi, nil)
		ops.ReLU(ff.Data)
		ops.LinearInto(y, ff, l.wo, nil)
		ops.Add(x.Data, y.Data)
	}
	ops.RMSNorm(x, m.encNorm, c.LayerNormEpsilon)
	return append(states, x), nil
}

// norm sets dst to RMSNorm(src) with weight w.
func (m *Model) norm(dst, src, w *tensor.Tensor) {
	copy(dst.Data, src.Data)
	ops.RMSNorm(dst, w, m.Config.LayerNormEpsilon)
}

// DecoderHiddenStates runs the decoder over decoder ids with the encoder
// output enc, attending to all of them at once (teacher forcing, no cache).
// It returns transformers' decoder_hidden_states (the input to each layer,
// then the final, normalized output) and the logits [len(ids), vocab].
func (m *Model) DecoderHiddenStates(enc *tensor.Tensor, ids []int) ([]*tensor.Tensor, *tensor.Tensor, error) {
	if err := m.checkIDs(ids); err != nil {
		return nil, nil, err
	}
	if m.gpu != nil {
		return m.gpu.decode(m, enc, ids)
	}
	c := m.Config
	n, encSeq, inner := len(ids), enc.Rows(), c.inner()
	y := ops.Gather(m.shared, ids)
	bias := positionBias(m.decBias, c, n, n, 0, false)
	var states []*tensor.Tensor
	h, z := tensor.New(n, c.DModel), tensor.New(n, c.DModel)
	qkv, q, ctx := tensor.New(n, 3*inner), tensor.New(n, inner), tensor.New(n, inner)
	kv := tensor.New(encSeq, 2*inner)
	ff := tensor.New(n, c.DFF)
	for _, l := range m.decoder {
		states = append(states, clone(y))
		m.norm(h, y, l.selfNorm)
		ops.LinearInto(qkv, h, l.selfQKV, nil)
		ops.AttentionStrided(ctx.Data, inner, qkv.Data, 3*inner, qkv.Data[inner:], 3*inner, qkv.Data[2*inner:], 3*inner,
			n, n, inner, ops.AttentionArgs{Heads: c.NumHeads, Scale: 1, Bias: bias, Causal: true})
		ops.LinearInto(z, ctx, l.selfOut, nil)
		ops.Add(y.Data, z.Data)

		m.norm(h, y, l.crossNorm)
		ops.LinearInto(q, h, l.crossQ, nil)
		ops.LinearInto(kv, enc, l.crossKV, nil)
		ops.AttentionStrided(ctx.Data, inner, q.Data, inner, kv.Data, 2*inner, kv.Data[inner:], 2*inner,
			n, encSeq, inner, ops.AttentionArgs{Heads: c.NumHeads, Scale: 1})
		ops.LinearInto(z, ctx, l.crossOut, nil)
		ops.Add(y.Data, z.Data)

		m.norm(h, y, l.ffNorm)
		ops.LinearInto(ff, h, l.wi, nil)
		ops.ReLU(ff.Data)
		ops.LinearInto(z, ff, l.wo, nil)
		ops.Add(y.Data, z.Data)
	}
	ops.RMSNorm(y, m.decNorm, c.LayerNormEpsilon)
	states = append(states, clone(y))
	return states, m.logits(y), nil
}

// logits applies the output layer to decoder outputs y (modified in place
// when the embeddings are tied, which scales y by d_model^-0.5 first).
func (m *Model) logits(y *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(y.Rows(), m.Config.VocabSize)
	m.logitsInto(out, y)
	return out
}

// logitsInto is logits writing into out [rows, vocab].
func (m *Model) logitsInto(out, y *tensor.Tensor) {
	if m.Config.tied() {
		s := float32(math.Pow(float64(m.Config.DModel), -0.5))
		for i := range y.Data {
			y.Data[i] *= s
		}
	}
	ops.MatMulPackedInto(out, y, m.lmHead)
}

// Generate encodes ids and greedily decodes up to maxNew tokens, always
// taking the most likely next token. It returns the generated ids, ending
// with the end-of-sequence id unless maxNew was reached first; the decoder
// start token is not included.
//
// Decoding keeps a cache of each layer's keys and values, so every step runs
// the decoder on the newest token only; the cross-attention keys and values
// are computed once from the encoder output.
func (m *Model) Generate(ids []int, maxNew int) ([]int, error) {
	enc, err := m.Encode(ids)
	if err != nil {
		return nil, err
	}
	st, err := m.newDecoderState(enc, maxNew)
	if err != nil {
		return nil, err
	}
	defer st.release()
	var out []int
	token := m.Config.DecoderStartTokenID
	for len(out) < maxNew {
		logits, err := st.step(token)
		if err != nil {
			return nil, err
		}
		token = ops.Argmax(logits)
		out = append(out, token)
		if token == m.Config.EOSTokenID {
			break
		}
	}
	return out, nil
}

// decoderState holds one sequence's decoding: the cross-attention keys and
// values (packed once from the encoder output), the self-attention cache,
// and buffers for one position, all reused across steps.
type decoderState struct {
	m     *Model
	cross []*ops.KVCache // per layer, the encoder output's keys and values
	self  []*ops.KVCache // per layer, one position per step so far
	pos   int

	y, h, z, qkv, q, ctx, ff *tensor.Tensor // one row each
	logits                   *tensor.Tensor // [1, vocab]
	bias                     []float32      // the step's position bias, [heads, pos+1]

	gpu     *gpuDecoder     // the GPU's caches and buffers, when the model runs there
	cleanup runtime.Cleanup // releases gpu when the state is unreachable
}

// newDecoderState prepares decoding over enc, with room for steps positions
// before the caches grow. Only the GPU fails.
func (m *Model) newDecoderState(enc *tensor.Tensor, steps int) (*decoderState, error) {
	c := m.Config
	inner, steps := c.inner(), max(steps, 1)
	if m.gpu != nil {
		g, err := m.gpu.newDecoder(m, enc, steps)
		if err != nil {
			return nil, err
		}
		st := &decoderState{m: m, gpu: g}
		st.cleanup = runtime.AddCleanup(st, (*gpuDecoder).release, g)
		return st, nil
	}
	st := &decoderState{
		m: m,
		y: tensor.New(1, c.DModel), h: tensor.New(1, c.DModel), z: tensor.New(1, c.DModel),
		qkv: tensor.New(1, 3*inner), q: tensor.New(1, inner), ctx: tensor.New(1, inner), ff: tensor.New(1, c.DFF),
		logits: tensor.New(1, c.VocabSize),
		bias:   make([]float32, 0, c.NumHeads*steps),
	}
	kv := tensor.New(enc.Rows(), 2*inner)
	for _, l := range m.decoder {
		ops.LinearInto(kv, enc, l.crossKV, nil)
		cross := ops.NewKVCache(c.NumHeads, c.DKV, enc.Rows())
		cross.Append(kv.Data, 2*inner, kv.Data[inner:], 2*inner, enc.Rows())
		st.cross = append(st.cross, cross)
		st.self = append(st.self, ops.NewKVCache(c.NumHeads, c.DKV, steps))
	}
	return st, nil
}

// release frees the state's GPU memory, if any, without waiting for the
// garbage collector. The state must not be used after.
func (st *decoderState) release() {
	if st.gpu != nil {
		st.cleanup.Stop()
		st.gpu.release()
		st.gpu = nil
	}
}

// step runs the decoder on token at the next position and returns the
// logits for the position after it. The slice is reused by the next step.
// Only the GPU fails.
func (st *decoderState) step(token int) ([]float32, error) {
	if st.gpu != nil {
		logits, err := st.gpu.step(token, st.pos)
		st.pos++
		return logits, err
	}
	m, c := st.m, st.m.Config
	inner, pos := c.inner(), st.pos
	copy(st.y.Data, m.shared.Row(token))
	// Keys 0..pos are all at or before this query, so no causal mask is
	// needed; the bias covers offsets -pos..0.
	st.bias = positionBiasInto(st.bias[:0], m.decBias, c, 1, pos+1, pos, false)
	for i, l := range m.decoder {
		m.norm(st.h, st.y, l.selfNorm)
		ops.LinearInto(st.qkv, st.h, l.selfQKV, nil)
		self := st.self[i]
		self.Append(st.qkv.Data[inner:], 3*inner, st.qkv.Data[2*inner:], 3*inner, 1)
		self.Attend(st.ctx.Data, inner, st.qkv.Data, 3*inner, 1, ops.AttentionArgs{Heads: c.NumHeads, Scale: 1, Bias: st.bias})
		ops.LinearInto(st.z, st.ctx, l.selfOut, nil)
		ops.Add(st.y.Data, st.z.Data)

		m.norm(st.h, st.y, l.crossNorm)
		ops.LinearInto(st.q, st.h, l.crossQ, nil)
		st.cross[i].Attend(st.ctx.Data, inner, st.q.Data, inner, 1, ops.AttentionArgs{Heads: c.NumHeads, Scale: 1})
		ops.LinearInto(st.z, st.ctx, l.crossOut, nil)
		ops.Add(st.y.Data, st.z.Data)

		m.norm(st.h, st.y, l.ffNorm)
		ops.LinearInto(st.ff, st.h, l.wi, nil)
		ops.ReLU(st.ff.Data)
		ops.LinearInto(st.z, st.ff, l.wo, nil)
		ops.Add(st.y.Data, st.z.Data)
	}
	st.pos++
	ops.RMSNorm(st.y, m.decNorm, c.LayerNormEpsilon)
	m.logitsInto(st.logits, st.y)
	return st.logits.Row(0), nil
}

// generateUncached is greedy generation that re-runs the decoder over the
// whole sequence for each new token. It is the reference Generate's cached
// decoding is tested against.
func (m *Model) generateUncached(ids []int, maxNew int) ([]int, error) {
	enc, err := m.Encode(ids)
	if err != nil {
		return nil, err
	}
	dec := []int{m.Config.DecoderStartTokenID}
	for len(dec)-1 < maxNew {
		_, logits, err := m.DecoderHiddenStates(enc, dec)
		if err != nil {
			return nil, err
		}
		next := ops.Argmax(logits.Row(logits.Rows() - 1))
		dec = append(dec, next)
		if next == m.Config.EOSTokenID {
			break
		}
	}
	return dec[1:], nil
}

// relativeBucket maps the offset from a query to a key (key position minus
// query position) to a bucket, exactly as transformers'
// T5Attention._relative_position_bucket, including its float32 arithmetic.
func relativeBucket(rel int, bidirectional bool, numBuckets, maxDistance int) int {
	bucket := 0
	if bidirectional {
		numBuckets /= 2
		if rel > 0 {
			bucket = numBuckets
		}
		rel = max(rel, -rel)
	} else {
		rel = -min(rel, 0)
	}
	maxExact := numBuckets / 2
	if rel < maxExact {
		return bucket + rel
	}
	scaled := float32(math.Log(float64(float32(rel)/float32(maxExact)))) /
		float32(math.Log(float64(maxDistance)/float64(maxExact))) *
		float32(numBuckets-maxExact)
	return bucket + min(maxExact+int(scaled), numBuckets-1)
}

// positionBias returns the bias [heads, seq, kvSeq] for queries at positions
// offset.. and keys at 0..kvSeq-1, from a bucket table [buckets, heads].
func positionBias(table *tensor.Tensor, c Config, seq, kvSeq, offset int, bidirectional bool) []float32 {
	return positionBiasInto(nil, table, c, seq, kvSeq, offset, bidirectional)
}

// positionBiasInto is positionBias reusing dst's storage when it is large
// enough.
func positionBiasInto(dst []float32, table *tensor.Tensor, c Config, seq, kvSeq, offset int, bidirectional bool) []float32 {
	heads := c.NumHeads
	bias := dst[:0]
	if need := heads * seq * kvSeq; cap(bias) >= need {
		bias = bias[:need]
	} else {
		bias = make([]float32, need, 2*need)
	}
	for i := range seq {
		for j := range kvSeq {
			b := relativeBucket(j-(offset+i), bidirectional, c.RelativeAttentionNumBuckets, c.RelativeAttentionMaxDistance)
			row := table.Row(b)
			for h := range heads {
				bias[(h*seq+i)*kvSeq+j] = row[h]
			}
		}
	}
	return bias
}

func clone(t *tensor.Tensor) *tensor.Tensor {
	c := tensor.New(t.Shape...)
	copy(c.Data, t.Data)
	return c
}

// loader fetches tensors by name, checks their shapes and keeps the first
// error so construction reads as a flat list.
type loader struct {
	weights *safetensors.File
	half    bool // keep F16 linear weights in fp16 panels
	err     error
}

func (l *loader) tensor(name string, shape ...int) *tensor.Tensor {
	if l.err != nil {
		return nil
	}
	t, err := l.weights.Tensor(name)
	if err != nil {
		l.err = err
		return nil
	}
	if len(t.Shape) != len(shape) {
		l.err = fmt.Errorf("tensor %s has shape %v, want %v", name, t.Shape, shape)
		return nil
	}
	for i := range shape {
		if t.Shape[i] != shape[i] {
			l.err = fmt.Errorf("tensor %s has shape %v, want %v", name, t.Shape, shape)
			return nil
		}
	}
	return t
}

// pack packs a linear weight, in fp16 panels when every source tensor is
// stored as F16 and fp16 panels are enabled.
func (l *loader) pack(w *tensor.Tensor, names ...string) *ops.Packed {
	if l.err != nil {
		return nil
	}
	if l.half {
		allHalf := true
		for _, name := range names {
			allHalf = allHalf && l.weights.DType(name) == "F16"
		}
		if allHalf {
			p, err := ops.PackHalf(w)
			if err != nil {
				l.err = err
				return nil
			}
			return p
		}
	}
	return ops.Pack(w)
}

func (l *loader) linear(name string, out, in int) *ops.Packed {
	return l.pack(l.tensor(name, out, in), name)
}

// stacked loads linear layers with the same input and stacks their outputs
// into one, so one product computes all of them.
func (l *loader) stacked(out, in int, names ...string) *ops.Packed {
	w := make([]float32, 0, len(names)*out*in)
	for _, name := range names {
		t := l.tensor(name, out, in)
		if l.err != nil {
			return nil
		}
		w = append(w, t.Data...)
	}
	return l.pack(tensor.FromData(w, len(names)*out, in), names...)
}
