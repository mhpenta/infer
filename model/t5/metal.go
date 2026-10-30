package t5

import (
	"errors"
	"fmt"
	"math"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/model/internal/gpu"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// The Metal path runs the same model on the GPU, when Options.Device or
// accel.Auto chooses it. Weights are copied once at load
// (model/internal/gpu) in their stored dtype; the relative position bias
// tables stay on the CPU, which turns them into each call's bias [heads,
// queries, keys] as before, and the GPU adds it to the scores.
//
// Encoding, and the teacher-forced decoder, are one command buffer each. A
// layer's output goes to a new slot (its residual adds read the previous
// slot), so the hidden states cost no extra round trips. Cached generation
// computes every layer's cross-attention keys and values once, then runs one
// command buffer per token. Results differ from the CPU path's within
// float32 rounding.

// gpuModel is a model's weights on the GPU.
type gpuModel struct {
	gpu              *accel.GPU
	buffers          []*metal.Buffer
	shared           metal.Ref // [vocab, d_model], in sharedType
	sharedType       metal.WeightType
	lmHead           metal.Weights
	encNorm, decNorm metal.Ref
	encoder          []gpuEncoderLayer
	decoder          []gpuDecoderLayer
}

type gpuEncoderLayer struct {
	attnNorm, ffNorm metal.Ref
	qkv, out, wi, wo metal.Weights
}

type gpuDecoderLayer struct {
	selfNorm, crossNorm, ffNorm metal.Ref
	selfQKV, selfOut            metal.Weights
	crossQ, crossKV, crossOut   metal.Weights
	wi, wo                      metal.Weights
}

// openGPU puts the model on the GPU, or explains why it can't.
func openGPU(c Config, weights *safetensors.File) (*gpuModel, error) {
	if c.DModel%4 != 0 || c.DFF%4 != 0 || c.DKV%32 != 0 || c.DKV > 256 {
		return nil, fmt.Errorf("the Metal kernels need d_model and d_ff that are multiples of 4 and a d_kv that is a multiple of 32 up to 256 (have %d, %d, %d)",
			c.DModel, c.DFF, c.DKV)
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
	dry := gpu.NewLoader(files, "", nil, false)
	loadGPU(dry, c)
	if dry.Err != nil {
		return nil, dry.Err
	}
	if err := gpu.Fits(g.Device, dry.Bytes); err != nil {
		return nil, err
	}
	l := gpu.NewLoader(files, "", g.Device, false)
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
	d, inner, ff := c.DModel, c.inner(), c.DFF
	linear := func(in, out int, names ...string) metal.Weights {
		outs := make([]int, len(names))
		for i := range outs {
			outs[i] = out
		}
		return l.Linear(in, outs, names, nil)
	}
	m := &gpuModel{
		encNorm: l.Vector("encoder.final_layer_norm.weight", d),
		decNorm: l.Vector("decoder.final_layer_norm.weight", d),
	}
	m.shared, m.sharedType = l.Table("shared.weight", c.VocabSize, d)
	m.lmHead = metal.Weights{W: m.shared, Type: m.sharedType, N: c.VocabSize, K: d}
	if !c.tied() {
		m.lmHead = linear(d, c.VocabSize, "lm_head.weight")
	}
	for i := range c.NumLayers {
		p := fmt.Sprintf("encoder.block.%d.layer.", i)
		m.encoder = append(m.encoder, gpuEncoderLayer{
			attnNorm: l.Vector(p+"0.layer_norm.weight", d),
			qkv:      linear(d, inner, p+"0.SelfAttention.q.weight", p+"0.SelfAttention.k.weight", p+"0.SelfAttention.v.weight"),
			out:      linear(inner, d, p+"0.SelfAttention.o.weight"),
			ffNorm:   l.Vector(p+"1.layer_norm.weight", d),
			wi:       linear(d, ff, p+"1.DenseReluDense.wi.weight"),
			wo:       linear(ff, d, p+"1.DenseReluDense.wo.weight"),
		})
	}
	for i := range c.NumDecoderLayers {
		p := fmt.Sprintf("decoder.block.%d.layer.", i)
		m.decoder = append(m.decoder, gpuDecoderLayer{
			selfNorm:  l.Vector(p+"0.layer_norm.weight", d),
			selfQKV:   linear(d, inner, p+"0.SelfAttention.q.weight", p+"0.SelfAttention.k.weight", p+"0.SelfAttention.v.weight"),
			selfOut:   linear(inner, d, p+"0.SelfAttention.o.weight"),
			crossNorm: l.Vector(p+"1.layer_norm.weight", d),
			crossQ:    linear(d, inner, p+"1.EncDecAttention.q.weight"),
			crossKV:   linear(d, inner, p+"1.EncDecAttention.k.weight", p+"1.EncDecAttention.v.weight"),
			crossOut:  linear(inner, d, p+"1.EncDecAttention.o.weight"),
			ffNorm:    l.Vector(p+"2.layer_norm.weight", d),
			wi:        linear(d, ff, p+"2.DenseReluDense.wi.weight"),
			wo:        linear(ff, d, p+"2.DenseReluDense.wo.weight"),
		})
	}
	return m
}

func (g *gpuModel) release() {
	gpu.Release(g.buffers)
	g.buffers = nil
}

// slots returns the i-th of a buffer's slots of rows×cols floats.
func slots(b *metal.Buffer, rows, cols int) func(i int) metal.Ref {
	return func(i int) metal.Ref { return b.At(4 * i * rows * cols) }
}

// states copies slots first..last of b out as tensors [rows, cols].
func states(b *metal.Buffer, first, last, rows, cols int) []*tensor.Tensor {
	var out []*tensor.Tensor
	for i := first; i <= last; i++ {
		t := tensor.New(rows, cols)
		copy(t.Data, metal.Slice[float32](b)[i*rows*cols:])
		out = append(out, t)
	}
	return out
}

// encode is Model.encode on the GPU.
func (g *gpuModel) encode(m *Model, ids []int, keepAll bool) ([]*tensor.Tensor, error) {
	c, k := m.Config, g.gpu.Kernels
	n, d, inner, layers := len(ids), c.DModel, c.inner(), len(g.encoder)
	s := gpu.NewScratch(g.gpu.Device)
	defer s.Release()
	idsB, xs := s.IDs(ids), s.Floats((layers+1)*n*d)
	h, qkv, ctx, ff := s.Floats(n*d), s.Floats(n*3*inner), s.Floats(n*inner), s.Floats(n*c.DFF)
	bias := s.Upload(positionBias(m.encBias, c, n, n, 0, true))
	cache, err := g.gpu.Device.NewKVCache(c.NumHeads, c.DKV, n)
	if err = errors.Join(s.Err, err); err != nil {
		return nil, err
	}
	defer cache.Release()
	slot := slots(xs, n, d)
	eps := float32(c.LayerNormEpsilon)
	attend := metal.AttendArgs{Heads: c.NumHeads, Scale: 1, Keys: n, Bias: bias.At(0)}

	cb := g.gpu.Device.NewCommandBuffer()
	k.Embed(cb, slot(0), g.shared, g.sharedType, idsB.At(0), c.VocabSize, n, d)
	for i, l := range g.encoder {
		x, next := slot(i), slot(i+1)
		k.RMSNorm(cb, h.At(0), x, l.attnNorm, d, n, 1, d, eps)
		k.Linear(cb, qkv.At(0), 3*inner, h.At(0), d, n, l.qkv)
		k.KVAppend(cb, cache, qkv.At(4*inner), qkv.At(8*inner), 3*inner, n, 0)
		k.Attend(cb, ctx.At(0), inner, qkv.At(0), 3*inner, cache, n, attend)
		k.Linear(cb, next, d, ctx.At(0), inner, n, l.out)
		k.Add(cb, next, x, n*d) // next = attention + x

		k.RMSNorm(cb, h.At(0), next, l.ffNorm, d, n, 1, d, eps)
		k.Linear(cb, ff.At(0), c.DFF, h.At(0), d, n, l.wi)
		k.ReLU(cb, ff.At(0), n*c.DFF)
		k.Linear(cb, h.At(0), d, ff.At(0), c.DFF, n, l.wo)
		k.Add(cb, next, h.At(0), n*d)
	}
	k.RMSNorm(cb, slot(layers), slot(layers), g.encNorm, d, n, 1, d, eps)
	if err := cb.Wait(); err != nil {
		return nil, err
	}
	first := layers
	if keepAll {
		first = 0
	}
	return states(xs, first, layers, n, d), nil
}

// decode is Model.DecoderHiddenStates on the GPU.
func (g *gpuModel) decode(m *Model, enc *tensor.Tensor, ids []int) ([]*tensor.Tensor, *tensor.Tensor, error) {
	c, k := m.Config, g.gpu.Kernels
	n, encSeq, d, inner, layers := len(ids), enc.Rows(), c.DModel, c.inner(), len(g.decoder)
	s := gpu.NewScratch(g.gpu.Device)
	defer s.Release()
	idsB, ys, encB := s.IDs(ids), s.Floats((layers+1)*n*d), s.Upload(enc.Data)
	h, qkv, q, ctx := s.Floats(n*d), s.Floats(n*3*inner), s.Floats(n*inner), s.Floats(n*inner)
	kv, ff, logits := s.Floats(encSeq*2*inner), s.Floats(n*c.DFF), s.Floats(n*c.VocabSize)
	bias := s.Upload(positionBias(m.decBias, c, n, n, 0, false))
	self, err := g.gpu.Device.NewKVCache(c.NumHeads, c.DKV, n)
	if err = errors.Join(s.Err, err); err != nil {
		return nil, nil, err
	}
	defer self.Release()
	cross, err := g.gpu.Device.NewKVCache(c.NumHeads, c.DKV, encSeq)
	if err != nil {
		return nil, nil, err
	}
	defer cross.Release()
	slot := slots(ys, n, d)
	eps := float32(c.LayerNormEpsilon)
	selfArgs := metal.AttendArgs{Heads: c.NumHeads, Scale: 1, Causal: true, Bias: bias.At(0)}
	crossArgs := metal.AttendArgs{Heads: c.NumHeads, Scale: 1, Keys: encSeq}

	cb := g.gpu.Device.NewCommandBuffer()
	k.Embed(cb, slot(0), g.shared, g.sharedType, idsB.At(0), c.VocabSize, n, d)
	for i, l := range g.decoder {
		y, next := slot(i), slot(i+1)
		k.RMSNorm(cb, h.At(0), y, l.selfNorm, d, n, 1, d, eps)
		k.Linear(cb, qkv.At(0), 3*inner, h.At(0), d, n, l.selfQKV)
		k.KVAppend(cb, self, qkv.At(4*inner), qkv.At(8*inner), 3*inner, n, 0)
		k.Attend(cb, ctx.At(0), inner, qkv.At(0), 3*inner, self, n, selfArgs)
		k.Linear(cb, next, d, ctx.At(0), inner, n, l.selfOut)
		k.Add(cb, next, y, n*d)

		k.RMSNorm(cb, h.At(0), next, l.crossNorm, d, n, 1, d, eps)
		k.Linear(cb, q.At(0), inner, h.At(0), d, n, l.crossQ)
		k.Linear(cb, kv.At(0), 2*inner, encB.At(0), d, encSeq, l.crossKV)
		k.KVAppend(cb, cross, kv.At(0), kv.At(4*inner), 2*inner, encSeq, 0)
		k.Attend(cb, ctx.At(0), inner, q.At(0), inner, cross, n, crossArgs)
		k.Linear(cb, h.At(0), d, ctx.At(0), inner, n, l.crossOut)
		k.Add(cb, next, h.At(0), n*d)

		k.RMSNorm(cb, h.At(0), next, l.ffNorm, d, n, 1, d, eps)
		k.Linear(cb, ff.At(0), c.DFF, h.At(0), d, n, l.wi)
		k.ReLU(cb, ff.At(0), n*c.DFF)
		k.Linear(cb, h.At(0), d, ff.At(0), c.DFF, n, l.wo)
		k.Add(cb, next, h.At(0), n*d)
	}
	out := slot(layers)
	k.RMSNorm(cb, out, out, g.decNorm, d, n, 1, d, eps)
	g.logits(m, cb, logits.At(0), out, h.At(0), n)
	if err := cb.Wait(); err != nil {
		return nil, nil, err
	}
	l := tensor.New(n, c.VocabSize)
	copy(l.Data, metal.Slice[float32](logits))
	return states(ys, 0, layers, n, d), l, nil
}

// logits records the output layer on y [rows, d_model] into out, scaling y
// by d_model^-0.5 into tmp first when the embeddings are tied, as
// Model.logitsInto does.
func (g *gpuModel) logits(m *Model, cb *metal.CommandBuffer, out, y, tmp metal.Ref, rows int) {
	c, k := m.Config, g.gpu.Kernels
	if c.tied() {
		k.Scale(cb, tmp, y, rows*c.DModel, float32(math.Pow(float64(c.DModel), -0.5)))
		y = tmp
	}
	k.Linear(cb, out, c.VocabSize, y, c.DModel, rows, g.lmHead)
}

// gpuDecoder is decoderState on the GPU: the cross-attention keys and
// values, computed once from the encoder output, the self-attention cache,
// and buffers for one position.
type gpuDecoder struct {
	g       *gpuModel
	m       *Model
	cross   []*metal.KVCache
	self    []*metal.KVCache
	encSeq  int
	scratch *gpu.Scratch

	id, y, h, qkv, q, ctx, ff, logits *metal.Buffer
	bias                              *metal.Buffer // [heads, capacity]
	biasCPU                           []float32
}

// newDecoder prepares decoding over enc with room for steps positions.
func (g *gpuModel) newDecoder(m *Model, enc *tensor.Tensor, steps int) (*gpuDecoder, error) {
	c, k := m.Config, g.gpu.Kernels
	d, inner, encSeq := c.DModel, c.inner(), enc.Rows()
	st := &gpuDecoder{g: g, m: m, encSeq: encSeq, scratch: gpu.NewScratch(g.gpu.Device)}
	s := st.scratch
	st.id, st.y, st.h, st.qkv = s.Floats(1), s.Floats(d), s.Floats(d), s.Floats(3*inner)
	st.q, st.ctx, st.ff, st.logits = s.Floats(inner), s.Floats(inner), s.Floats(c.DFF), s.Floats(c.VocabSize)
	once := gpu.NewScratch(g.gpu.Device)
	defer once.Release()
	encB, kv := once.Upload(enc.Data), once.Floats(encSeq*2*inner)
	if err := errors.Join(s.Err, once.Err, st.grow(steps)); err != nil {
		st.release()
		return nil, err
	}
	cb := g.gpu.Device.NewCommandBuffer()
	for _, l := range g.decoder {
		cross, err := g.gpu.Device.NewKVCache(c.NumHeads, c.DKV, encSeq)
		if err != nil {
			cb.Wait()
			st.release()
			return nil, err
		}
		st.cross = append(st.cross, cross)
		k.Linear(cb, kv.At(0), 2*inner, encB.At(0), d, encSeq, l.crossKV)
		k.KVAppend(cb, cross, kv.At(0), kv.At(4*inner), 2*inner, encSeq, 0)
	}
	if err := cb.Wait(); err != nil {
		st.release()
		return nil, err
	}
	return st, nil
}

// grow makes room for capacity positions in the self-attention caches,
// keeping what they hold. The GPU is idle between steps and the memory is
// shared, so the CPU copies.
func (st *gpuDecoder) grow(capacity int) error {
	c := st.m.Config
	have := 0
	if len(st.self) > 0 {
		have = st.self[0].Capacity
	}
	if capacity <= have {
		return nil
	}
	capacity = max(capacity, 2*have)
	bias, err := st.g.gpu.Device.NewBuffer(4 * c.NumHeads * capacity)
	if err != nil {
		return err
	}
	var caches []*metal.KVCache
	for i := range st.g.decoder {
		kv, err := st.g.gpu.Device.NewKVCache(c.NumHeads, c.DKV, capacity)
		if err != nil {
			bias.Release()
			for _, made := range caches {
				made.Release()
			}
			return err
		}
		if i < len(st.self) {
			old := st.self[i]
			for _, pair := range [][2]*metal.Buffer{{old.Keys, kv.Keys}, {old.Values, kv.Values}} {
				from, to := metal.Slice[float32](pair[0]), metal.Slice[float32](pair[1])
				for h := range c.NumHeads {
					copy(to[h*capacity*c.DKV:], from[h*have*c.DKV:(h+1)*have*c.DKV])
				}
			}
		}
		caches = append(caches, kv)
	}
	for _, old := range st.self {
		old.Release()
	}
	if st.bias != nil {
		st.bias.Release()
	}
	st.self, st.bias = caches, bias
	return nil
}

func (st *gpuDecoder) release() {
	for _, kv := range append(st.cross, st.self...) {
		kv.Release()
	}
	st.cross, st.self = nil, nil
	if st.bias != nil {
		st.bias.Release()
		st.bias = nil
	}
	st.scratch.Release()
}

// step runs the decoder on token at position pos and returns the logits for
// the position after it, valid until the next step.
func (st *gpuDecoder) step(token, pos int) ([]float32, error) {
	c, g := st.m.Config, st.g
	k := g.gpu.Kernels
	d, inner := c.DModel, c.inner()
	if err := st.grow(pos + 1); err != nil {
		return nil, err
	}
	metal.Slice[int32](st.id)[0] = int32(token)
	st.biasCPU = positionBiasInto(st.biasCPU[:0], st.m.decBias, c, 1, pos+1, pos, false)
	copy(metal.Slice[float32](st.bias), st.biasCPU)
	eps := float32(c.LayerNormEpsilon)
	y, h, ctx := st.y.At(0), st.h.At(0), st.ctx.At(0)
	selfArgs := metal.AttendArgs{Heads: c.NumHeads, Scale: 1, Causal: true, Offset: pos, Bias: st.bias.At(0)}
	crossArgs := metal.AttendArgs{Heads: c.NumHeads, Scale: 1, Keys: st.encSeq}

	cb := g.gpu.Device.NewCommandBuffer()
	k.Embed(cb, y, g.shared, g.sharedType, st.id.At(0), c.VocabSize, 1, d)
	for i, l := range g.decoder {
		k.RMSNorm(cb, h, y, l.selfNorm, d, 1, 1, d, eps)
		k.Linear(cb, st.qkv.At(0), 3*inner, h, d, 1, l.selfQKV)
		k.KVAppend(cb, st.self[i], st.qkv.At(4*inner), st.qkv.At(8*inner), 3*inner, 1, pos)
		k.Attend(cb, ctx, inner, st.qkv.At(0), 3*inner, st.self[i], 1, selfArgs)
		k.Linear(cb, h, d, ctx, inner, 1, l.selfOut)
		k.Add(cb, y, h, d)

		k.RMSNorm(cb, h, y, l.crossNorm, d, 1, 1, d, eps)
		k.Linear(cb, st.q.At(0), inner, h, d, 1, l.crossQ)
		k.Attend(cb, ctx, inner, st.q.At(0), inner, st.cross[i], 1, crossArgs)
		k.Linear(cb, h, d, ctx, inner, 1, l.crossOut)
		k.Add(cb, y, h, d)

		k.RMSNorm(cb, h, y, l.ffNorm, d, 1, 1, d, eps)
		k.Linear(cb, st.ff.At(0), c.DFF, h, d, 1, l.wi)
		k.ReLU(cb, st.ff.At(0), c.DFF)
		k.Linear(cb, h, d, st.ff.At(0), c.DFF, 1, l.wo)
		k.Add(cb, y, h, d)
	}
	k.RMSNorm(cb, y, y, g.decNorm, d, 1, 1, d, eps)
	g.logits(st.m, cb, st.logits.At(0), y, h, 1)
	if err := cb.Wait(); err != nil {
		return nil, err
	}
	return metal.Slice[float32](st.logits), nil
}
