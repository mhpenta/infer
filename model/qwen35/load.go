package qwen35

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

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

// scale loads a zero-centered RMSNorm weight w and returns its scale, 1+w,
// as Qwen3.5's RMSNorm computes it in float32.
func (l *loader) scale(name string, n int) *tensor.Tensor {
	t := l.tensor(name, n)
	if t == nil {
		return nil
	}
	for i, w := range t.Data {
		t.Data[i] = 1 + w
	}
	return t
}

// pack packs a linear weight as int8 when the loader quantizes, and
// otherwise as bf16 panels when its source tensors are all bf16, float32
// otherwise.
func (l *loader) pack(w *tensor.Tensor, names ...string) *ops.Packed {
	if l.err != nil {
		return nil
	}
	if l.int8 {
		return ops.PackInt8(w)
	}
	bf16 := true
	for _, name := range names {
		bf16 = bf16 && l.dtype(name) == "BF16"
	}
	if !bf16 {
		return ops.Pack(w)
	}
	p, err := ops.PackBF16(w)
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
	var rows [][]float32
	for i, name := range names {
		t := l.tensor(name, outs[i], in)
		if l.err != nil {
			return nil
		}
		rows = append(rows, t.Data)
	}
	return l.stack(in, rows, names)
}

func (l *loader) stack(in int, parts [][]float32, names []string) *ops.Packed {
	w := slices.Concat(parts...)
	return l.pack(tensor.FromData(w, len(w)/in, in), names...)
}

// attentionIn loads a full-attention layer's input projections as one: the
// queries of every head, then the output gate of every head, then k and v.
// q_proj's rows alternate one head's queries with that head's gate.
func (l *loader) attentionIn(c Config, p string) *ops.Packed {
	h, hd := c.HiddenSize, c.HeadDim
	names := []string{p + "q_proj.weight", p + "k_proj.weight", p + "v_proj.weight"}
	qg := l.tensor(names[0], 2*c.qDim(), h)
	k := l.tensor(names[1], c.kvDim(), h)
	v := l.tensor(names[2], c.kvDim(), h)
	if l.err != nil {
		return nil
	}
	q, gate := make([]float32, 0, c.qDim()*h), make([]float32, 0, c.qDim()*h)
	for head := range c.NumAttentionHeads {
		rows := qg.Data[head*2*hd*h : (head+1)*2*hd*h]
		q = append(q, rows[:hd*h]...)
		gate = append(gate, rows[hd*h:]...)
	}
	return l.stack(h, [][]float32{q, gate, k.Data, v.Data}, names)
}
