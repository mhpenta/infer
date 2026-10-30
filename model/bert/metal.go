package bert

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/model/internal/gpu"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// The Metal path runs the same encoder on the GPU, when Options.Device or
// accel.Auto chooses it. Weights are copied once at load
// (model/internal/gpu), keeping their stored dtype (float32 for MiniLM, fp16
// for mxbai); position and token-type embeddings and the LayerNorms are
// float32. A forward pass is one command buffer: the embeddings, then each
// layer's products (with their biases), attention over every position, and
// LayerNorms with the residual add fused in. Every layer writes its output
// to a separate slot, so HiddenStates costs no extra round trips. Results
// differ from the CPU path's within float32 rounding.

// gpuModel is a model's weights on the GPU.
type gpuModel struct {
	gpu       *accel.GPU
	buffers   []*metal.Buffer
	word      metal.Ref // [vocab, hidden], in wordType
	wordType  metal.WeightType
	pos, typ  metal.Ref // float32 [max positions, hidden], [types, hidden]
	embNorm   gpuNorm
	layers    []gpuLayer
	mu        sync.Mutex
	workspace []*gpuWork // free workspaces
}

type gpuNorm struct{ w, b metal.Ref }

type gpuLayer struct {
	qkv, attnOut, intermediate, output metal.Weights
	attnNorm, outNorm                  gpuNorm
}

// openGPU puts the model on the GPU, or explains why it can't.
func openGPU(c Config, weights *safetensors.File, prefix string) (*gpuModel, error) {
	headDim := c.HiddenSize / c.NumAttentionHeads
	if c.HiddenSize%4 != 0 || c.IntermediateSize%4 != 0 || headDim%32 != 0 || headDim > 256 {
		return nil, fmt.Errorf("the Metal kernels need hidden and intermediate sizes that are multiples of 4 and a head dimension that is a multiple of 32 up to 256 (have %d, %d, %d)",
			c.HiddenSize, c.IntermediateSize, headDim)
	}
	g, err := accel.OpenMetal()
	if err != nil {
		return nil, err
	}
	files := func(name string) *safetensors.File {
		if weights.Has(name) {
			return weights
		}
		return nil
	}
	dry := gpu.NewLoader(files, prefix, nil, false)
	loadGPU(dry, c)
	if dry.Err != nil {
		return nil, dry.Err
	}
	if err := gpu.Fits(g.Device, dry.Bytes); err != nil {
		return nil, err
	}
	l := gpu.NewLoader(files, prefix, g.Device, false)
	m := loadGPU(l, c)
	if l.Err != nil {
		gpu.Release(l.Buffers)
		return nil, l.Err
	}
	m.gpu, m.buffers = g, l.Buffers
	return m, nil
}

// loadGPU loads the model's weights with l.
func loadGPU(l *gpu.Loader, c Config) *gpuModel {
	h, ff := c.HiddenSize, c.IntermediateSize
	norm := func(name string) gpuNorm { return gpuNorm{l.Vector(name+".weight", h), l.Vector(name+".bias", h)} }
	linear := func(in, out int, names ...string) metal.Weights {
		var weights, biases []string
		var outs []int
		for _, name := range names {
			weights, biases, outs = append(weights, name+".weight"), append(biases, name+".bias"), append(outs, out)
		}
		return l.Linear(in, outs, weights, biases)
	}
	m := &gpuModel{
		pos:     l.Matrix("embeddings.position_embeddings.weight", c.MaxPositionEmbeddings, h),
		typ:     l.Matrix("embeddings.token_type_embeddings.weight", c.TypeVocabSize, h),
		embNorm: norm("embeddings.LayerNorm"),
	}
	m.word, m.wordType = l.Table("embeddings.word_embeddings.weight", c.VocabSize, h)
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("encoder.layer.%d.", i)
		m.layers = append(m.layers, gpuLayer{
			qkv:          linear(h, h, p+"attention.self.query", p+"attention.self.key", p+"attention.self.value"),
			attnOut:      linear(h, h, p+"attention.output.dense"),
			attnNorm:     norm(p + "attention.output.LayerNorm"),
			intermediate: linear(h, ff, p+"intermediate.dense"),
			output:       linear(ff, h, p+"output.dense"),
			outNorm:      norm(p + "output.LayerNorm"),
		})
	}
	return m
}

// release frees the weights and every workspace.
func (g *gpuModel) release() {
	gpu.Release(g.buffers)
	g.buffers = nil
	g.mu.Lock()
	defer g.mu.Unlock()
	for _, w := range g.workspace {
		w.release()
	}
	g.workspace = nil
}

// gpuWork is one forward pass's buffers, for up to rows positions.
type gpuWork struct {
	rows                             int
	ids, states, qkv, ctx, attn, ffn *metal.Buffer
	cache                            *metal.KVCache
}

func (w *gpuWork) release() {
	for _, b := range []*metal.Buffer{w.ids, w.states, w.qkv, w.ctx, w.attn, w.ffn} {
		if b != nil {
			b.Release()
		}
	}
	if w.cache != nil {
		w.cache.Release()
	}
}

// work returns a free workspace for rows positions, or makes one.
func (g *gpuModel) work(c Config, rows int) (*gpuWork, error) {
	g.mu.Lock()
	for i, w := range g.workspace {
		if w.rows >= rows {
			g.workspace = append(g.workspace[:i], g.workspace[i+1:]...)
			g.mu.Unlock()
			return w, nil
		}
	}
	g.mu.Unlock()
	dev, h := g.gpu.Device, c.HiddenSize
	w := &gpuWork{rows: rows}
	var errs []error
	alloc := func(floats int) *metal.Buffer {
		b, err := dev.NewBuffer(4 * floats)
		errs = append(errs, err)
		return b
	}
	w.ids, w.states = alloc(rows), alloc((len(g.layers)+1)*rows*h)
	w.qkv, w.ctx, w.attn, w.ffn = alloc(rows*3*h), alloc(rows*h), alloc(rows*h), alloc(rows*c.IntermediateSize)
	var err error
	w.cache, err = dev.NewKVCache(c.NumAttentionHeads, h/c.NumAttentionHeads, rows)
	if err = errors.Join(append(errs, err)...); err != nil {
		w.release()
		return nil, err
	}
	return w, nil
}

func (g *gpuModel) done(w *gpuWork) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.workspace = append(g.workspace, w)
}

// run returns the embeddings and every layer's output, or only the last
// output.
func (g *gpuModel) run(c Config, ids []int, keepAll bool) ([]*tensor.Tensor, error) {
	k := g.gpu.Kernels
	n, h, ff := len(ids), c.HiddenSize, c.IntermediateSize
	w, err := g.work(c, n)
	if err != nil {
		return nil, err
	}
	defer g.done(w)
	for i, id := range ids {
		metal.Slice[int32](w.ids)[i] = int32(id)
	}
	slot := func(i int) metal.Ref { return w.states.At(4 * i * n * h) }
	eps := float32(c.LayerNormEps)
	attend := metal.AttendArgs{Heads: c.NumAttentionHeads, Scale: float32(1 / math.Sqrt(float64(h/c.NumAttentionHeads))), Keys: n}
	attn, ctx, qkv, ffn := w.attn.At(0), w.ctx.At(0), w.qkv.At(0), w.ffn.At(0)

	cb := g.gpu.Device.NewCommandBuffer()
	x := slot(0)
	k.Embed(cb, x, g.word, g.wordType, w.ids.At(0), c.VocabSize, n, h)
	k.Add(cb, x, g.pos, n*h)     // positions 0..n-1
	k.AddRow(cb, x, g.typ, n, h) // token type 0 throughout
	k.LayerNorm(cb, x, metal.Ref{}, g.embNorm.w, g.embNorm.b, n, h, eps)
	for i, l := range g.layers {
		x, next := slot(i), slot(i+1)
		k.Linear(cb, qkv, 3*h, x, h, n, l.qkv)
		k.KVAppend(cb, w.cache, w.qkv.At(4*h), w.qkv.At(8*h), 3*h, n, 0)
		k.Attend(cb, ctx, h, qkv, 3*h, w.cache, n, attend)
		k.Linear(cb, attn, h, ctx, h, n, l.attnOut)
		k.LayerNorm(cb, attn, x, l.attnNorm.w, l.attnNorm.b, n, h, eps)
		k.Linear(cb, ffn, ff, attn, h, n, l.intermediate)
		k.GELU(cb, ffn, n*ff)
		k.Linear(cb, next, h, ffn, ff, n, l.output)
		k.LayerNorm(cb, next, attn, l.outNorm.w, l.outNorm.b, n, h, eps)
	}
	if err := cb.Wait(); err != nil {
		return nil, err
	}
	first := len(g.layers)
	if keepAll {
		first = 0
	}
	var states []*tensor.Tensor
	for i := first; i <= len(g.layers); i++ {
		t := tensor.New(n, h)
		copy(t.Data, metal.Slice[float32](w.states)[i*n*h:(i+1)*n*h])
		states = append(states, t)
	}
	return states, nil
}
