// Package llama runs decoder-only transformers of the Llama family
// (LlamaForCausalLM checkpoints such as MiniCPM5) from Hugging Face weights:
// RMSNorm, grouped-query attention with rotary position embeddings, a SwiGLU
// feed-forward, and generation with a KV cache: greedy, or sampled with the
// sample package.
//
// Every product runs in float32. Weights stored as bf16 stay bf16 in memory
// (packed with ops.PackBF16 and widened in the kernels, so results are the
// same as with float32 weights), which halves memory and the bandwidth each
// generated token needs. LoadWith can instead quantize the weights to int8
// (Options.Int8), halving them again at a small, measured cost in accuracy.
package llama

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/sample"
	"github.com/mhpenta/infer/tensor"
)

// Config is the subset of config.json the model uses.
type Config struct {
	ModelType         string          `json:"model_type"`
	HiddenSize        int             `json:"hidden_size"`
	IntermediateSize  int             `json:"intermediate_size"`
	NumHiddenLayers   int             `json:"num_hidden_layers"`
	NumAttentionHeads int             `json:"num_attention_heads"`
	NumKeyValueHeads  int             `json:"num_key_value_heads"`
	HeadDim           int             `json:"head_dim"`
	VocabSize         int             `json:"vocab_size"`
	RMSNormEps        float64         `json:"rms_norm_eps"`
	RopeTheta         float64         `json:"rope_theta"`
	RopeScaling       json.RawMessage `json:"rope_scaling"`
	HiddenAct         string          `json:"hidden_act"`
	AttentionBias     bool            `json:"attention_bias"`
	MLPBias           bool            `json:"mlp_bias"`
	TieWordEmbeddings bool            `json:"tie_word_embeddings"`
	EOSTokenID        json.RawMessage `json:"eos_token_id"`

	// EOS are the end tokens, from eos_token_id (one id or a list).
	EOS []int `json:"-"`
}

// withDefaults fills the fields transformers' LlamaConfig defaults.
func (c Config) withDefaults() (Config, error) {
	if c.NumKeyValueHeads == 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.HeadDim == 0 && c.NumAttentionHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = 1e-6
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000
	}
	if c.HiddenAct == "" {
		c.HiddenAct = "silu"
	}
	if len(c.EOSTokenID) > 0 && string(c.EOSTokenID) != "null" {
		var one int
		if err := json.Unmarshal(c.EOSTokenID, &one); err == nil {
			c.EOS = []int{one}
		} else if err := json.Unmarshal(c.EOSTokenID, &c.EOS); err != nil {
			return c, fmt.Errorf("eos_token_id %s is neither an id nor a list", c.EOSTokenID)
		}
	}
	return c, nil
}

// validate rejects configurations this package does not implement rather
// than running them wrong.
func (c Config) validate() error {
	switch {
	case c.ModelType != "llama":
		return fmt.Errorf("model_type %q is not llama", c.ModelType)
	case c.HiddenAct != "silu":
		return fmt.Errorf("hidden_act %q is not supported", c.HiddenAct)
	case c.AttentionBias || c.MLPBias:
		return errors.New("attention and MLP biases are not supported")
	case len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null":
		return fmt.Errorf("rope_scaling %s is not supported", c.RopeScaling)
	case c.HiddenSize <= 0 || c.IntermediateSize <= 0 || c.NumHiddenLayers <= 0 || c.VocabSize <= 0:
		return errors.New("config is missing sizes")
	case c.NumAttentionHeads <= 0 || c.NumKeyValueHeads <= 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0:
		return fmt.Errorf("%d attention heads do not split into %d key/value heads", c.NumAttentionHeads, c.NumKeyValueHeads)
	case c.HeadDim <= 0 || c.HeadDim%2 != 0:
		return fmt.Errorf("head_dim %d is not even", c.HeadDim)
	}
	return nil
}

func (c Config) qDim() int  { return c.NumAttentionHeads * c.HeadDim }
func (c Config) kvDim() int { return c.NumKeyValueHeads * c.HeadDim }

// Model is a loaded decoder. It is safe for concurrent use: generation keeps
// its state (the KV cache) per call.
type Model struct {
	Config Config

	files  []*safetensors.File
	embed  embedding
	layers []layer
	norm   *tensor.Tensor // final RMSNorm weight
	lmHead *ops.Packed    // output layer [vocab, hidden]
	rope   *ops.RoPE
}

type layer struct {
	attnNorm, ffNorm *tensor.Tensor
	qkv              *ops.Packed // q, k and v projections stacked
	out              *ops.Packed
	gateUp           *ops.Packed // gate and up projections stacked
	down             *ops.Packed
}

// embedding is the token embedding table: bf16 rows widened on lookup when
// stored as bf16 (it is the largest tensor), float32 otherwise.
type embedding struct {
	cols int
	bf16 []uint16
	f32  *tensor.Tensor
}

func (e *embedding) row(id int, dst []float32) {
	if e.bf16 == nil {
		copy(dst, e.f32.Row(id))
		return
	}
	for j, h := range e.bf16[id*e.cols : (id+1)*e.cols] {
		dst[j] = math.Float32frombits(uint32(h) << 16)
	}
}

// Options control how LoadWith prepares a model.
type Options struct {
	// Int8 quantizes the linear layers and the output layer to int8 with one
	// scale per output (ops.PackInt8) instead of keeping the checkpoint's
	// precision. The token embeddings keep theirs: generation reads one row
	// of them per token.
	Int8 bool
}

// Load reads config.json and the weights (model.safetensors, or the shards
// model.safetensors.index.json lists) from dir, keeping the weights in the
// checkpoint's precision.
func Load(dir string) (*Model, error) {
	return LoadWith(dir, Options{})
}

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("llama: parse config.json: %w", err)
	}
	if config, err = config.withDefaults(); err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	files, byName, err := openWeights(dir)
	if err != nil {
		return nil, fmt.Errorf("llama: %w", err)
	}
	m, err := newModel(config, byName, opts)
	if err != nil {
		for _, f := range files {
			f.Close()
		}
		return nil, fmt.Errorf("llama: %w", err)
	}
	m.files = files
	return m, nil
}

// openWeights opens model.safetensors, or every shard of a sharded
// checkpoint, and maps each tensor name to its file.
func openWeights(dir string) ([]*safetensors.File, map[string]*safetensors.File, error) {
	byName := map[string]*safetensors.File{}
	single := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(single); err == nil {
		f, err := safetensors.Open(single)
		if err != nil {
			return nil, nil, err
		}
		for _, name := range f.Names() {
			byName[name] = f
		}
		return []*safetensors.File{f}, byName, nil
	}
	raw, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		return nil, nil, fmt.Errorf("no model.safetensors or index: %w", err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, nil, fmt.Errorf("parse model.safetensors.index.json: %w", err)
	}
	var files []*safetensors.File
	shards := map[string]*safetensors.File{}
	names := make([]string, 0, len(index.WeightMap))
	for name := range index.WeightMap {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		shard := index.WeightMap[name]
		f, ok := shards[shard]
		if !ok {
			if f, err = safetensors.Open(filepath.Join(dir, filepath.Base(shard))); err != nil {
				for _, open := range files {
					open.Close()
				}
				return nil, nil, err
			}
			shards[shard] = f
			files = append(files, f)
		}
		byName[name] = f
	}
	return files, byName, nil
}

func newModel(c Config, byName map[string]*safetensors.File, opts Options) (*Model, error) {
	l := loader{files: byName, int8: opts.Int8}
	m := &Model{Config: c, rope: ops.NewRoPE(c.HeadDim, c.RopeTheta)}
	h, ff := c.HiddenSize, c.IntermediateSize
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		m.layers = append(m.layers, layer{
			attnNorm: l.tensor(p+"input_layernorm.weight", h),
			ffNorm:   l.tensor(p+"post_attention_layernorm.weight", h),
			qkv: l.stacked(h, []string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"},
				[]int{c.qDim(), c.kvDim(), c.kvDim()}),
			out:    l.linear(p+"self_attn.o_proj.weight", h, c.qDim()),
			gateUp: l.stacked(h, []string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, []int{ff, ff}),
			down:   l.linear(p+"mlp.down_proj.weight", h, ff),
		})
	}
	m.norm = l.tensor("model.norm.weight", h)
	const embedName = "model.embed_tokens.weight"
	embed := l.tensor(embedName, c.VocabSize, h)
	if c.TieWordEmbeddings {
		m.lmHead = l.pack(embed, embedName)
	} else {
		m.lmHead = l.linear("lm_head.weight", c.VocabSize, h)
	}
	if l.err != nil {
		return nil, l.err
	}
	m.embed = embedding{cols: h, f32: embed}
	if l.dtype(embedName) == "BF16" {
		m.embed = embedding{cols: h, bf16: make([]uint16, len(embed.Data))}
		for i, v := range embed.Data {
			m.embed.bf16[i] = uint16(math.Float32bits(v) >> 16) // exact: the value came from bf16
		}
	}
	return m, nil
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
		return errors.New("llama: no input ids")
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.VocabSize {
			return fmt.Errorf("llama: id %d outside the vocabulary of %d", id, m.Config.VocabSize)
		}
	}
	return nil
}

// HiddenStates runs the whole sequence (no cache reuse) and returns
// num_hidden_layers+1 states [len(ids), hidden]: the input to each layer
// (the first is the embeddings), then the final normalized output, as
// transformers' output_hidden_states.
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

// LogitsInto sets out [rows, vocab] to the output layer applied to hidden
// states y [rows, hidden] (final, normalized outputs).
func (m *Model) LogitsInto(out, y *tensor.Tensor) {
	ops.MatMulPackedInto(out, y, m.lmHead)
}

// Generate decodes greedily after ids (the prompt, already tokenized with
// any special tokens) and returns the new tokens: up to maxNew, ending with
// an end token (Config.EOS) if one is produced.
func (m *Model) Generate(ids []int, maxNew int) ([]int, error) {
	var out []int
	err := m.GenerateFunc(ids, maxNew, func(token int, _ []float32) bool {
		out = append(out, token)
		return true
	})
	return out, err
}

// GenerateFunc is Generate calling fn with each new token and the logits it
// was chosen from (reused by the next step) as soon as it is chosen; fn
// returning false stops generation.
func (m *Model) GenerateFunc(ids []int, maxNew int, fn func(token int, logits []float32) bool) error {
	return m.GenerateWith(ids, maxNew, nil, fn)
}

// GenerateWith is GenerateFunc choosing each token with s instead of
// greedily (a nil s is greedy): s sees the prompt and every token generated
// so far, for its repetition penalty. fn receives the model's logits, before
// any of s's processing.
func (m *Model) GenerateWith(ids []int, maxNew int, s *sample.Sampler, fn func(token int, logits []float32) bool) error {
	if err := m.checkIDs(ids); err != nil {
		return err
	}
	st := m.newState(len(ids) + maxNew)
	logits := tensor.New(1, m.Config.VocabSize)
	seq := slices.Grow(slices.Clone(ids), maxNew)
	y := st.forward(ids, nil)
	for range maxNew {
		m.LogitsInto(logits, tensor.FromData(y.Row(y.Rows()-1), 1, y.Cols()))
		var token int
		if s == nil {
			token = ops.Argmax(logits.Data)
		} else {
			token = s.Next(logits.Data, seq)
		}
		seq = append(seq, token)
		if !fn(token, logits.Data) || slices.Contains(m.Config.EOS, token) {
			return nil
		}
		y = st.forward([]int{token}, nil)
	}
	return nil
}

// state is one sequence being decoded: its position and per-layer caches.
type state struct {
	m      *Model
	pos    int
	caches []*ops.KVCache
}

// newState returns an empty sequence with cache room for capacity positions.
func (m *Model) newState(capacity int) *state {
	st := &state{m: m}
	for range m.layers {
		st.caches = append(st.caches, ops.NewKVCache(m.Config.NumKeyValueHeads, m.Config.HeadDim, capacity))
	}
	return st
}

// forward runs ids at the next positions, appending their keys and values
// to the caches, and returns the final normalized hidden states
// [len(ids), hidden]. keep, if not nil, sees the input to every layer and
// then the final output.
func (st *state) forward(ids []int, keep func(*tensor.Tensor)) *tensor.Tensor {
	m, c := st.m, st.m.Config
	n, qDim, kvDim := len(ids), c.qDim(), c.kvDim()
	width := qDim + 2*kvDim
	x := tensor.New(n, c.HiddenSize)
	for i, id := range ids {
		m.embed.row(id, x.Row(i))
	}
	h, y := tensor.New(n, c.HiddenSize), tensor.New(n, c.HiddenSize)
	qkv, ctx := tensor.New(n, width), tensor.New(n, qDim)
	gateUp, act := tensor.New(n, 2*c.IntermediateSize), tensor.New(n, c.IntermediateSize)
	rot := m.rope.Rotation(st.pos, n)
	args := ops.AttentionArgs{
		Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads,
		Scale:  float32(1 / math.Sqrt(float64(c.HeadDim))),
		Causal: true, Offset: st.pos,
	}
	for i, l := range m.layers {
		if keep != nil {
			keep(x)
		}
		rmsNormInto(h, x, l.attnNorm, c.RMSNormEps)
		ops.LinearInto(qkv, h, l.qkv, nil)
		rot.Apply(qkv.Data, width, c.NumAttentionHeads)
		rot.Apply(qkv.Data[qDim:], width, c.NumKeyValueHeads)
		cache := st.caches[i]
		cache.Append(qkv.Data[qDim:], width, qkv.Data[qDim+kvDim:], width, n)
		cache.Attend(ctx.Data, qDim, qkv.Data, width, n, args)
		ops.LinearInto(y, ctx, l.out, nil)
		ops.Add(x.Data, y.Data)

		rmsNormInto(h, x, l.ffNorm, c.RMSNormEps)
		ops.LinearInto(gateUp, h, l.gateUp, nil)
		ops.SwiGLUInto(act, gateUp)
		ops.LinearInto(y, act, l.down, nil)
		ops.Add(x.Data, y.Data)
	}
	st.pos += n
	ops.RMSNorm(x, m.norm, c.RMSNormEps)
	if keep != nil {
		keep(x)
	}
	return x
}

// rmsNormInto sets dst to RMSNorm(src) with weight w.
func rmsNormInto(dst, src, w *tensor.Tensor, eps float64) {
	copy(dst.Data, src.Data)
	ops.RMSNorm(dst, w, eps)
}

// loader reads tensors by name, remembering the first error.
type loader struct {
	files map[string]*safetensors.File
	int8  bool // quantize linear weights to int8
	err   error
}

func (l *loader) dtype(name string) string {
	if f, ok := l.files[name]; ok {
		return f.DType(name)
	}
	return ""
}

func (l *loader) tensor(name string, shape ...int) *tensor.Tensor {
	if l.err != nil {
		return nil
	}
	f, ok := l.files[name]
	if !ok {
		l.err = fmt.Errorf("tensor %s is missing", name)
		return nil
	}
	t, err := f.Tensor(name)
	if err != nil {
		l.err = err
		return nil
	}
	if !slices.Equal(t.Shape, shape) {
		l.err = fmt.Errorf("tensor %s has shape %v, want %v", name, t.Shape, shape)
		return nil
	}
	return t
}

// pack packs a linear weight as int8 when the loader quantizes, and
// otherwise in the storage of its source tensors: bf16 or fp16 panels when
// they all share that dtype, float32 otherwise.
func (l *loader) pack(w *tensor.Tensor, names ...string) *ops.Packed {
	if l.err != nil {
		return nil
	}
	if l.int8 {
		return ops.PackInt8(w)
	}
	dtype := l.dtype(names[0])
	for _, name := range names[1:] {
		if l.dtype(name) != dtype {
			dtype = ""
		}
	}
	var p *ops.Packed
	var err error
	switch dtype {
	case "BF16":
		p, err = ops.PackBF16(w)
	case "F16":
		p, err = ops.PackHalf(w)
	default:
		p = ops.Pack(w)
	}
	if err != nil {
		l.err = err
	}
	return p
}

// linear loads an nn.Linear weight [out, in].
func (l *loader) linear(name string, out, in int) *ops.Packed {
	return l.pack(l.tensor(name, out, in), name)
}

// stacked loads linear layers over the same input, with outs[i] outputs
// each, and stacks them into one so one product computes all of them.
func (l *loader) stacked(in int, names []string, outs []int) *ops.Packed {
	total := 0
	for _, o := range outs {
		total += o
	}
	w := make([]float32, 0, total*in)
	for i, name := range names {
		t := l.tensor(name, outs[i], in)
		if l.err != nil {
			return nil
		}
		w = append(w, t.Data...)
	}
	return l.pack(tensor.FromData(w, total, in), names...)
}
