// Package bert implements the BERT encoder for inference.
//
// The architecture is post-LN: word + position + token-type embeddings are
// summed and layer-normalized, then each layer applies self-attention and a
// GELU feed-forward block, each followed by a residual add and LayerNorm.
// Weights use the Hugging Face BertModel tensor names.
//
// On Apple silicon the encoder can run on the GPU instead (Options.Device,
// metal.go): by default it does when the GPU is available and suits the
// model.
package bert

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// Config is the subset of a Hugging Face BERT config.json the encoder needs.
type Config struct {
	VocabSize             int     `json:"vocab_size"`
	HiddenSize            int     `json:"hidden_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	IntermediateSize      int     `json:"intermediate_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	TypeVocabSize         int     `json:"type_vocab_size"`
	LayerNormEps          float64 `json:"layer_norm_eps"`
	HiddenAct             string  `json:"hidden_act"`
	PositionEmbeddingType string  `json:"position_embedding_type"`
}

func (c Config) validate() error {
	switch {
	case c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0:
		return fmt.Errorf("hidden_size, num_hidden_layers and num_attention_heads must be positive")
	case c.HiddenSize%c.NumAttentionHeads != 0:
		return fmt.Errorf("hidden_size %d is not divisible by %d heads", c.HiddenSize, c.NumAttentionHeads)
	case c.HiddenAct != "gelu":
		return fmt.Errorf("hidden_act %q is not supported (want gelu)", c.HiddenAct)
	case c.PositionEmbeddingType != "" && c.PositionEmbeddingType != "absolute":
		return fmt.Errorf("position_embedding_type %q is not supported (want absolute)", c.PositionEmbeddingType)
	}
	return nil
}

// Model is a loaded BERT encoder. It is safe for concurrent use.
type Model struct {
	Config Config

	wordEmb, posEmb, typeEmb *tensor.Tensor
	embNorm                  layerNorm
	layers                   []layer
	weights                  *safetensors.File

	// workspaces recycles the activation buffers of a forward pass.
	workspaces sync.Pool

	gpu    *gpuModel // the weights on the GPU, when the model runs there
	device string
}

type linear struct {
	weight *ops.Packed
	bias   *tensor.Tensor
}

type layerNorm struct{ weight, bias *tensor.Tensor }

type layer struct {
	qkv                  linear // query, key and value projections, stacked
	attnOut              linear
	attnNorm             layerNorm
	intermediate, output linear
	outNorm              layerNorm
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

// Load reads config.json and model.safetensors from a model directory. The
// weights stay memory-mapped until Close.
func Load(dir string) (*Model, error) {
	return LoadWith(dir, Options{})
}

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (*Model, error) {
	device, err := accel.Resolve(opts.Device)
	if err != nil {
		return nil, fmt.Errorf("bert: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("bert: %w", err)
	}
	var config Config
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("bert: parse config.json: %w", err)
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("bert: %w", err)
	}
	weights, err := safetensors.Open(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("bert: %w", err)
	}
	var m *Model
	if device != accel.CPU {
		var g *gpuModel
		if g, err = openGPU(config, weights, prefixOf(weights)); err == nil {
			m = &Model{Config: config, weights: weights, gpu: g, device: fmt.Sprintf("%s (%s)", accel.Metal, g.gpu.Device.Name())}
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
		return nil, fmt.Errorf("bert: %w", err)
	}
	return m, nil
}

// prefixOf returns the prefix of the encoder's tensor names: checkpoints
// saved from task models (BertForMaskedLM, ...) nest it under "bert.".
func prefixOf(weights *safetensors.File) string {
	if !weights.Has("embeddings.word_embeddings.weight") && weights.Has("bert.embeddings.word_embeddings.weight") {
		return "bert."
	}
	return ""
}

func newModel(config Config, weights *safetensors.File) (*Model, error) {
	l := loader{weights: weights, prefix: prefixOf(weights), half: useHalfPanels()}
	h := config.HiddenSize
	m := &Model{
		Config:  config,
		wordEmb: l.tensor("embeddings.word_embeddings.weight", config.VocabSize, h),
		posEmb:  l.tensor("embeddings.position_embeddings.weight", config.MaxPositionEmbeddings, h),
		typeEmb: l.tensor("embeddings.token_type_embeddings.weight", config.TypeVocabSize, h),
		embNorm: l.layerNorm("embeddings.LayerNorm", h),
		weights: weights,
		device:  accel.CPU,
	}
	for i := range config.NumHiddenLayers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.layers = append(m.layers, layer{
			qkv:          l.stacked(h, h, p+"attention.self.query", p+"attention.self.key", p+"attention.self.value"),
			attnOut:      l.linear(p+"attention.output.dense", h, h),
			attnNorm:     l.layerNorm(p+"attention.output.LayerNorm", h),
			intermediate: l.linear(p+"intermediate.dense", config.IntermediateSize, h),
			output:       l.linear(p+"output.dense", h, config.IntermediateSize),
			outNorm:      l.layerNorm(p+"output.LayerNorm", h),
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

// Forward runs the encoder on one sequence of token ids and returns the last
// hidden state, [len(ids), hidden_size].
func (m *Model) Forward(ids []int) (*tensor.Tensor, error) {
	states, err := m.run(ids, false)
	if err != nil {
		return nil, err
	}
	return states[len(states)-1], nil
}

// HiddenStates is Forward but returns every hidden state: the embedding
// output followed by each layer's output, like output_hidden_states=True.
func (m *Model) HiddenStates(ids []int) ([]*tensor.Tensor, error) {
	return m.run(ids, true)
}

func (m *Model) run(ids []int, keepAll bool) ([]*tensor.Tensor, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("bert: empty input")
	}
	if len(ids) > m.Config.MaxPositionEmbeddings {
		return nil, fmt.Errorf("bert: %d tokens exceeds max_position_embeddings %d", len(ids), m.Config.MaxPositionEmbeddings)
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.VocabSize {
			return nil, fmt.Errorf("bert: token id %d outside vocab of %d", id, m.Config.VocabSize)
		}
	}

	if m.gpu != nil {
		return m.gpu.run(m.Config, ids, keepAll)
	}
	ws := m.workspace(len(ids))
	defer m.workspaces.Put(ws)
	x, next := ws.x, ws.next
	m.embed(ids, x)
	var states []*tensor.Tensor
	if keepAll {
		states = append(states, clone(x))
	}
	for _, l := range m.layers {
		m.layerForward(l, x, next, ws)
		x, next = next, x
		if keepAll {
			states = append(states, clone(x))
		}
	}
	if !keepAll {
		states = append(states, clone(x))
	}
	return states, nil
}

// workspace holds the activations of one forward pass. Each layer reads x
// and writes next, and the two swap, so a pass allocates nothing per layer.
type workspace struct {
	buf                                 []float32
	x, next, qkv, context, attn, hidden *tensor.Tensor
}

// workspace returns a pooled workspace sized for seq tokens. Return it to
// m.workspaces when the pass is done; results must be copied out first.
func (m *Model) workspace(seq int) *workspace {
	h, f := m.Config.HiddenSize, m.Config.IntermediateSize
	ws, _ := m.workspaces.Get().(*workspace)
	if ws == nil {
		ws = &workspace{}
	}
	if need := seq * (7*h + f); cap(ws.buf) < need {
		ws.buf = make([]float32, need)
	}
	rest := ws.buf
	take := func(cols int) *tensor.Tensor {
		t := tensor.FromData(rest[:seq*cols:seq*cols], seq, cols)
		rest = rest[seq*cols:]
		return t
	}
	ws.x, ws.next, ws.qkv, ws.context, ws.attn = take(h), take(h), take(3*h), take(h), take(h)
	ws.hidden = take(f)
	return ws
}

func clone(t *tensor.Tensor) *tensor.Tensor {
	c := tensor.New(t.Shape...)
	copy(c.Data, t.Data)
	return c
}

// embed writes the sum of word, position and token-type embeddings (single
// segment, so type 0 throughout), layer-normalized, into x.
func (m *Model) embed(ids []int, x *tensor.Tensor) {
	typeRow := m.typeEmb.Row(0)
	for i, id := range ids {
		row := x.Row(i)
		copy(row, m.wordEmb.Row(id))
		ops.Add(row, m.posEmb.Row(i))
		ops.Add(row, typeRow)
	}
	ops.LayerNorm(x, m.embNorm.weight, m.embNorm.bias, m.Config.LayerNormEps)
}

// layerForward runs one encoder layer from x into out, using ws for the
// intermediate activations.
func (m *Model) layerForward(l layer, x, out *tensor.Tensor, ws *workspace) {
	headDim := m.Config.HiddenSize / m.Config.NumAttentionHeads
	ops.LinearInto(ws.qkv, x, l.qkv.weight, l.qkv.bias)
	ops.AttentionQKVInto(ws.context, ws.qkv, m.Config.NumAttentionHeads, float32(1/math.Sqrt(float64(headDim))))

	ops.LinearInto(ws.attn, ws.context, l.attnOut.weight, l.attnOut.bias)
	ops.AddLayerNorm(ws.attn, x, l.attnNorm.weight, l.attnNorm.bias, m.Config.LayerNormEps)

	ops.LinearInto(ws.hidden, ws.attn, l.intermediate.weight, l.intermediate.bias)
	ops.GELU(ws.hidden.Data)
	ops.LinearInto(out, ws.hidden, l.output.weight, l.output.bias)
	ops.AddLayerNorm(out, ws.attn, l.outNorm.weight, l.outNorm.bias, m.Config.LayerNormEps)
}

// useHalfPanels reports whether to keep F16 weights in fp16 panels. It is a
// variable so tests can load a model both ways and compare.
var useHalfPanels = ops.HalfPanels

// loader fetches tensors by name, checks their shapes and keeps the first
// error so construction reads as a flat list.
type loader struct {
	weights *safetensors.File
	prefix  string
	half    bool // pack F16 linear weights into fp16 panels
	err     error
}

// pack packs a linear weight, in fp16 panels when every source tensor is
// stored as F16 and fp16 panels are enabled: the values are fp16 already, so
// nothing is lost, and products read half the memory.
func (l *loader) pack(w *tensor.Tensor, names ...string) *ops.Packed {
	if l.half {
		allHalf := true
		for _, name := range names {
			allHalf = allHalf && l.weights.DType(l.prefix+name) == "F16"
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

func (l *loader) tensor(name string, shape ...int) *tensor.Tensor {
	if l.err != nil {
		return nil
	}
	t, err := l.weights.Tensor(l.prefix + name)
	if err != nil {
		l.err = err
		return nil
	}
	if !equalShape(t.Shape, shape) {
		l.err = fmt.Errorf("tensor %s has shape %v, want %v", l.prefix+name, t.Shape, shape)
		return nil
	}
	return t
}

// linear loads an nn.Linear and packs its weight for the matmul kernel. The
// packed copy replaces the mapped original, which is not kept.
func (l *loader) linear(name string, out, in int) linear {
	w := l.tensor(name+".weight", out, in)
	b := l.tensor(name+".bias", out)
	if l.err != nil {
		return linear{}
	}
	return linear{weight: l.pack(w, name+".weight"), bias: b}
}

// stacked loads several nn.Linear layers with the same input and stacks their
// outputs into one: rows of the weights and entries of the biases are
// concatenated in order, so one product computes all of them.
func (l *loader) stacked(out, in int, names ...string) linear {
	w := make([]float32, 0, len(names)*out*in)
	b := make([]float32, 0, len(names)*out)
	weightNames := make([]string, len(names))
	for i, name := range names {
		weightNames[i] = name + ".weight"
		wt, bt := l.tensor(name+".weight", out, in), l.tensor(name+".bias", out)
		if l.err != nil {
			return linear{}
		}
		w, b = append(w, wt.Data...), append(b, bt.Data...)
	}
	n := len(names) * out
	packed := l.pack(tensor.FromData(w, n, in), weightNames...)
	if l.err != nil {
		return linear{}
	}
	return linear{weight: packed, bias: tensor.FromData(b, n)}
}

func (l *loader) layerNorm(name string, size int) layerNorm {
	return layerNorm{weight: l.tensor(name+".weight", size), bias: l.tensor(name+".bias", size)}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
