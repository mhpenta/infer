// Package gpu is what the models' Metal paths share: loading weights from
// safetensors files into GPU buffers, ready for package metal's kernels.
//
// Linear layers keep the dtype the checkpoint stores (bf16, fp16 or float32),
// or, with int8, are quantized as on the CPU (ops.QuantizeInt8, the same
// integers and scales as ops.PackInt8). Layers over the same input can be
// stacked into one, so one product computes all of them. Each model records
// its own forward pass from these pieces.
package gpu

import (
	"fmt"
	"slices"

	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/safetensors"
)

// Loader copies tensors into GPU buffers, remembering the first error, so a
// model's loading reads as a flat list. A dry loader only checks every
// tensor's presence, shape and dtype and adds up the bytes, so a model can
// decide whether it fits before anything is allocated.
type Loader struct {
	files  func(name string) *safetensors.File
	prefix string
	dev    *metal.Device // nil for a dry loader
	int8   bool

	// Buffers are the buffers loaded, which every Ref returned points into.
	Buffers []*metal.Buffer
	// Bytes is the total size of what was loaded, or would have been.
	Bytes int64
	// Err is the first error.
	Err error
}

// NewLoader returns a loader reading tensors from the files files returns
// for a name (nil if no file has it), each name prefixed with prefix, and
// quantizing linear layers when int8 is set. A nil dev makes a dry loader.
func NewLoader(files func(name string) *safetensors.File, prefix string, dev *metal.Device, int8 bool) *Loader {
	return &Loader{files: files, prefix: prefix, dev: dev, int8: int8}
}

// Release releases the buffers loaded.
func Release(buffers []*metal.Buffer) {
	for _, b := range buffers {
		b.Release()
	}
}

// alloc returns a new buffer of n bytes, or nil for a dry loader or after
// an error.
func (l *Loader) alloc(n int) *metal.Buffer {
	l.Bytes += int64(n)
	if l.dev == nil || l.Err != nil {
		return nil
	}
	b, err := l.dev.NewBuffer(n)
	if err != nil {
		l.Err = err
		return nil
	}
	l.Buffers = append(l.Buffers, b)
	return b
}

// file returns the file holding name, after checking its shape.
func (l *Loader) file(name string, shape ...int) *safetensors.File {
	if l.Err != nil {
		return nil
	}
	f := l.files(l.prefix + name)
	if f == nil {
		l.Err = fmt.Errorf("tensor %s is missing", l.prefix+name)
		return nil
	}
	if got := f.Shape(l.prefix + name); !slices.Equal(got, shape) {
		l.Err = fmt.Errorf("tensor %s has shape %v, want %v", l.prefix+name, got, shape)
		return nil
	}
	return f
}

// stored returns the weight type of a tensor's dtype.
func stored(dtype string) (metal.WeightType, bool) {
	switch dtype {
	case "BF16":
		return metal.BF16, true
	case "F16":
		return metal.F16, true
	case "F32":
		return metal.F32, true
	}
	return 0, false
}

// Linear loads linear layers over the same in inputs, outs[i] outputs each,
// stacked into one [Σouts, in] layer, with biases (one per weight, or nil
// for none). The weights stay in their stored dtype, which they must share,
// or are quantized to int8 row by row (so quantizing then stacking gives the
// stack's integers).
func (l *Loader) Linear(in int, outs []int, weights, biases []string) metal.Weights {
	total := 0
	var t metal.WeightType
	for i, name := range weights {
		f := l.file(name, outs[i], in)
		if f == nil {
			return metal.Weights{}
		}
		ti, ok := stored(f.DType(l.prefix + name))
		if !ok || i > 0 && ti != t {
			l.Err = fmt.Errorf("tensor %s is %s; the GPU takes bf16, fp16 or float32 layers, all of one type when stacked", l.prefix+name, f.DType(l.prefix+name))
			return metal.Weights{}
		}
		t = ti
		total += outs[i]
	}
	w := metal.Weights{Type: t, N: total, K: in}
	if l.int8 {
		w.Type = metal.Int8
		w.W, w.Scale = l.quantized(in, total, weights)
	} else {
		w.W = l.raw(t.Size()*total*in, weights)
	}
	if biases != nil {
		w.Bias = l.vectors(biases, outs)
	}
	return w
}

// raw loads tensors' stored bytes back to back into one buffer of size
// bytes.
func (l *Loader) raw(size int, names []string) metal.Ref {
	b := l.alloc(size)
	if b == nil {
		return metal.Ref{}
	}
	off := 0
	for _, name := range names {
		raw, err := l.files(l.prefix + name).Bytes(l.prefix + name)
		if err != nil {
			l.Err = err
			return metal.Ref{}
		}
		off += copy(b.Bytes()[off:], raw)
	}
	return b.At(0)
}

// quantized loads tensors, rows rows of in values in all, quantized to int8,
// and their scales.
func (l *Loader) quantized(in, rows int, names []string) (q, scale metal.Ref) {
	qb, sb := l.alloc(rows*in), l.alloc(4*rows)
	if qb == nil || sb == nil {
		return metal.Ref{}, metal.Ref{}
	}
	row := 0
	for _, name := range names {
		t, err := l.files(l.prefix + name).Tensor(l.prefix + name)
		if err != nil {
			l.Err = err
			return metal.Ref{}, metal.Ref{}
		}
		qs, ss := ops.QuantizeInt8(t)
		copy(metal.Slice[int8](qb)[row*in:], qs)
		copy(metal.Slice[float32](sb)[row:], ss)
		row += t.Rows()
	}
	return qb.At(0), sb.At(0)
}

// Vector loads a vector of n values as float32.
func (l *Loader) Vector(name string, n int) metal.Ref {
	return l.vectors([]string{name}, []int{n})
}

// vectors loads vectors of ns[i] values back to back as float32.
func (l *Loader) vectors(names []string, ns []int) metal.Ref {
	total := 0
	for i, name := range names {
		l.file(name, ns[i])
		total += ns[i]
	}
	return l.floats(names, total)
}

// Matrix loads a [rows, cols] tensor as float32.
func (l *Loader) Matrix(name string, rows, cols int) metal.Ref {
	l.file(name, rows, cols)
	return l.floats([]string{name}, rows*cols)
}

// floats loads tensors, total values in all, back to back as float32.
func (l *Loader) floats(names []string, total int) metal.Ref {
	b := l.alloc(4 * total)
	if b == nil {
		return metal.Ref{}
	}
	off := 0
	for _, name := range names {
		t, err := l.files(l.prefix + name).Tensor(l.prefix + name)
		if err != nil {
			l.Err = err
			return metal.Ref{}
		}
		off += copy(metal.Slice[float32](b)[off:], t.Data)
	}
	return b.At(0)
}

// Table loads an embedding table [rows, cols] in its stored dtype, which
// must be bf16, fp16 or float32.
func (l *Loader) Table(name string, rows, cols int) (metal.Ref, metal.WeightType) {
	f := l.file(name, rows, cols)
	if f == nil {
		return metal.Ref{}, 0
	}
	t, ok := stored(f.DType(l.prefix + name))
	if !ok {
		l.Err = fmt.Errorf("tensor %s is %s; the GPU takes bf16, fp16 or float32 tables", l.prefix+name, f.DType(l.prefix+name))
		return metal.Ref{}, 0
	}
	return l.raw(t.Size()*rows*cols, []string{name}), t
}

// Fits returns an error unless bytes of weights leave a quarter of the
// GPU's recommended working set for everything else (caches, activations).
func Fits(dev *metal.Device, bytes int64) error {
	if limit := dev.RecommendedMaxWorkingSetSize(); bytes > limit*3/4 {
		return fmt.Errorf("%.1f GB of weights leave too little of the GPU's %.1f GB working set",
			float64(bytes)/1e9, float64(limit)/1e9)
	}
	return nil
}

// Scratch allocates buffers for one call and releases them together. After
// a failed allocation, Err is set and later allocations return nil, so a
// caller allocates everything, then checks Err once.
type Scratch struct {
	dev     *metal.Device
	buffers []*metal.Buffer
	Err     error
}

// NewScratch returns a Scratch allocating on dev.
func NewScratch(dev *metal.Device) *Scratch { return &Scratch{dev: dev} }

// Floats returns a buffer of n float32 values (or n int32 values).
func (s *Scratch) Floats(n int) *metal.Buffer {
	if s.Err != nil {
		return nil
	}
	b, err := s.dev.NewBuffer(4 * max(n, 1))
	if err != nil {
		s.Err = err
		return nil
	}
	s.buffers = append(s.buffers, b)
	return b
}

// Upload returns a buffer holding v.
func (s *Scratch) Upload(v []float32) *metal.Buffer {
	b := s.Floats(len(v))
	if b != nil {
		copy(metal.Slice[float32](b), v)
	}
	return b
}

// IDs returns a buffer holding ids as int32.
func (s *Scratch) IDs(ids []int) *metal.Buffer {
	b := s.Floats(len(ids))
	if b != nil {
		for i, id := range ids {
			metal.Slice[int32](b)[i] = int32(id)
		}
	}
	return b
}

// Release releases every buffer.
func (s *Scratch) Release() {
	Release(s.buffers)
	s.buffers = nil
}
