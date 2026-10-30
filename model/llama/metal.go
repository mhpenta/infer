package llama

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

// The Metal path runs the same decoder on the GPU (package metal), when
// Options.Device or accel.Auto chooses it.
//
// Weights are copied once at load into GPU buffers (model/internal/gpu),
// row-major as the checkpoint stores them, with q|k|v and gate|up stacked by
// placing them back to back; nothing is packed. They keep their stored
// dtype (bf16, fp16 or float32), or, with Options.Int8, the linear layers
// and the output layer are quantized as on the CPU (ops.QuantizeInt8, the
// same integers and scales as ops.PackInt8); the embeddings keep their
// dtype either way. Each forward pass is one command buffer
// holding every layer's dispatches, then the final norm and, when
// generating, the output layer on the last position, so a decoding step
// costs one round trip to the GPU. Prompts run in chunks of gpuChunk
// positions.
//
// A prompt's products and attention run on the simdgroup matrix units
// (GemmBF16 and GemmInt8, and Attend's tiled kernel), a decoding step's on the one-row
// kernels, and the two sum in different orders: one-row kernels on the
// matrix units are 5-7 times slower. So unlike on the CPU, the keys and
// values a Session reuses from decoding steps are not bit for bit the ones
// a fresh prompt computes, and its logits match a fresh run's within float32
// rounding rather than exactly. Results differ from the CPU path's within
// float32 rounding too: the GPU sums in other orders.

// gpuChunk is the most positions one command buffer runs.
const gpuChunk = 512

// gpuModel is a model's weights on the GPU.
type gpuModel struct {
	gpu       *accel.GPU
	buffers   []*metal.Buffer // every Ref below points into one of these
	embed     metal.Ref       // [vocab, hidden], in embedType
	embedType metal.WeightType
	norm      metal.Ref
	lmHead    metal.Weights
	layers    []gpuLayer
}

type gpuLayer struct {
	attnNorm, ffNorm metal.Ref
	qNorm, kNorm     metal.Ref     // Buf is nil without per-head norms
	qkv, out         metal.Weights // [qDim+2·kvDim, hidden], [hidden, qDim]
	gateUp, down     metal.Weights // [2·ff, hidden], [hidden, ff]
}

// openGPU puts the model on the GPU, or explains why it can't: no Metal
// device, shapes the kernels do not take, weights in a dtype the GPU does
// not take (it takes bf16, fp16 and float32, and int8 quantizes any of
// them), or weights too large for the GPU's working set.
func openGPU(c Config, byName map[string]*safetensors.File, opts Options) (*gpuModel, error) {
	if c.HiddenSize%4 != 0 || c.IntermediateSize%4 != 0 || c.HeadDim%32 != 0 || c.HeadDim > 256 {
		return nil, fmt.Errorf("the Metal kernels need hidden and intermediate sizes that are multiples of 4 and a head dimension that is a multiple of 32 up to 256 (have %d, %d, %d)",
			c.HiddenSize, c.IntermediateSize, c.HeadDim)
	}
	g, err := accel.OpenMetal()
	if err != nil {
		return nil, err
	}
	files := func(name string) *safetensors.File { return byName[name] }
	// A dry run checks every tensor and adds up the bytes before anything
	// is allocated.
	dry := gpu.NewLoader(files, "", nil, opts.Int8)
	loadGPU(dry, c, opts.Int8)
	if dry.Err != nil {
		return nil, dry.Err
	}
	if err := gpu.Fits(g.Device, dry.Bytes); err != nil {
		return nil, err
	}
	l := gpu.NewLoader(files, "", g.Device, opts.Int8)
	m := loadGPU(l, c, opts.Int8)
	if l.Err != nil {
		gpu.Release(l.Buffers)
		return nil, l.Err
	}
	m.gpu, m.buffers = g, l.Buffers
	return m, nil
}

func (g *gpuModel) release() {
	gpu.Release(g.buffers)
	g.buffers = nil
}

// loadGPU loads the model's weights with l.
func loadGPU(l *gpu.Loader, c Config, int8 bool) *gpuModel {
	h, ff := c.HiddenSize, c.IntermediateSize
	m := &gpuModel{}
	for i := range c.NumHiddenLayers {
		p := fmt.Sprintf("model.layers.%d.", i)
		layer := gpuLayer{
			attnNorm: l.Vector(p+"input_layernorm.weight", h),
			ffNorm:   l.Vector(p+"post_attention_layernorm.weight", h),
			qkv: l.Linear(h, []int{c.qDim(), c.kvDim(), c.kvDim()},
				[]string{p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, nil),
			out:    l.Linear(c.qDim(), []int{h}, []string{p + "self_attn.o_proj.weight"}, nil),
			gateUp: l.Linear(h, []int{ff, ff}, []string{p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, nil),
			down:   l.Linear(ff, []int{h}, []string{p + "mlp.down_proj.weight"}, nil),
		}
		if c.qkNorm() {
			layer.qNorm, layer.kNorm = l.Vector(p+"self_attn.q_norm.weight", c.HeadDim), l.Vector(p+"self_attn.k_norm.weight", c.HeadDim)
		}
		m.layers = append(m.layers, layer)
	}
	m.norm = l.Vector("model.norm.weight", h)
	const embedName = "model.embed_tokens.weight"
	m.embed, m.embedType = l.Table(embedName, c.VocabSize, h)
	switch {
	case !c.TieWordEmbeddings:
		m.lmHead = l.Linear(h, []int{c.VocabSize}, []string{"lm_head.weight"}, nil)
	case int8:
		m.lmHead = l.Linear(h, []int{c.VocabSize}, []string{embedName}, nil)
	default:
		m.lmHead = metal.Weights{W: m.embed, Type: m.embedType, N: c.VocabSize, K: h}
	}
	return m
}

// gpuState is one sequence's caches and scratch buffers on the GPU,
// allocated on first use and grown as needed.
type gpuState struct {
	m                                                     *Model
	caches                                                []*metal.KVCache
	capacity                                              int // positions the caches are to hold when first allocated
	rows                                                  int // rows the scratch buffers hold
	x, h, y, qkv, ctx, gateUp, act, ids, cos, sin, logits *metal.Buffer
}

func (s *gpuState) release() {
	for _, kv := range s.caches {
		kv.Release()
	}
	s.caches = nil
	s.releaseScratch()
	if s.logits != nil {
		s.logits.Release()
		s.logits = nil
	}
}

func (s *gpuState) releaseScratch() {
	for _, b := range []*metal.Buffer{s.x, s.h, s.y, s.qkv, s.ctx, s.gateUp, s.act, s.ids, s.cos, s.sin} {
		if b != nil {
			b.Release()
		}
	}
	s.x, s.h, s.y, s.qkv, s.ctx, s.gateUp, s.act, s.ids, s.cos, s.sin = nil, nil, nil, nil, nil, nil, nil, nil, nil, nil
	s.rows = 0
}

// reserve makes room for rows rows of scratch and positions cached
// positions, keeping what the caches hold.
func (s *gpuState) reserve(rows, positions int) error {
	c, dev := s.m.Config, s.m.gpu.gpu.Device
	if s.logits == nil {
		var err error
		if s.logits, err = dev.NewBuffer(4 * c.VocabSize); err != nil {
			return err
		}
	}
	if rows > s.rows {
		s.releaseScratch()
		width := c.qDim() + 2*c.kvDim()
		var errs []error
		alloc := func(floats int) *metal.Buffer {
			b, err := dev.NewBuffer(4 * floats)
			errs = append(errs, err)
			return b
		}
		s.x, s.h, s.y = alloc(rows*c.HiddenSize), alloc(rows*c.HiddenSize), alloc(rows*c.HiddenSize)
		s.qkv, s.ctx = alloc(rows*width), alloc(rows*c.qDim())
		s.gateUp, s.act = alloc(rows*2*c.IntermediateSize), alloc(rows*c.IntermediateSize)
		s.ids, s.cos, s.sin = alloc(rows), alloc(rows*c.HeadDim/2), alloc(rows*c.HeadDim/2)
		if err := errors.Join(errs...); err != nil {
			return err
		}
		s.rows = rows
	}
	have := s.capacity
	if len(s.caches) > 0 {
		have = s.caches[0].Capacity
	}
	if len(s.caches) > 0 && positions <= have {
		return nil
	}
	capacity := max(have, positions)
	if len(s.caches) > 0 {
		capacity = max(2*have, positions)
	}
	return s.grow(capacity)
}

// grow replaces the caches with ones of capacity positions, copying what
// the old ones hold. The GPU is idle between forward passes and the memory
// is shared, so the CPU copies.
func (s *gpuState) grow(capacity int) error {
	c := s.m.Config
	caches := make([]*metal.KVCache, len(s.m.gpu.layers))
	for i := range caches {
		kv, err := s.m.gpu.gpu.Device.NewKVCache(c.NumKeyValueHeads, c.HeadDim, capacity)
		if err != nil {
			for _, made := range caches[:i] {
				made.Release()
			}
			return err
		}
		caches[i] = kv
	}
	for i, old := range s.caches {
		for _, pair := range [][2]*metal.Buffer{{old.Keys, caches[i].Keys}, {old.Values, caches[i].Values}} {
			from, to := metal.Slice[float32](pair[0]), metal.Slice[float32](pair[1])
			for h := range c.NumKeyValueHeads {
				copy(to[h*capacity*c.HeadDim:], from[h*old.Capacity*c.HeadDim:(h+1)*old.Capacity*c.HeadDim])
			}
		}
		old.Release()
	}
	s.caches = caches
	return nil
}

// run runs ids at positions pos.., appending their keys and values to the
// caches. With hidden, it returns the final normalized hidden states
// [len(ids), hidden] in Go memory; with logits, it leaves the last
// position's logits in s.logits. keep, if not nil, sees the input to every
// layer and then the final output, all rows at once.
func (s *gpuState) run(pos int, ids []int, keep func(*tensor.Tensor), hidden, logits bool) (*tensor.Tensor, error) {
	c := s.m.Config
	var out *tensor.Tensor
	if hidden {
		out = tensor.New(len(ids), c.HiddenSize)
	}
	chunk := gpuChunk
	if keep != nil {
		chunk = len(ids)
	}
	for start := 0; start < len(ids); start += chunk {
		part := ids[start:min(start+chunk, len(ids))]
		last := start+len(part) == len(ids)
		if err := s.chunk(pos+start, part, keep, logits && last); err != nil {
			return nil, err
		}
		if hidden {
			copy(out.Data[start*c.HiddenSize:], metal.Slice[float32](s.x)[:len(part)*c.HiddenSize])
		}
	}
	return out, nil
}

// chunk runs ids at positions pos.. in one command buffer (one per layer
// with keep, which reads every layer's input).
func (s *gpuState) chunk(pos int, ids []int, keep func(*tensor.Tensor), logits bool) error {
	c, g := s.m.Config, s.m.gpu
	k := g.gpu.Kernels
	n, h, ff := len(ids), c.HiddenSize, c.IntermediateSize
	qDim, kvDim := c.qDim(), c.kvDim()
	width := qDim + 2*kvDim
	if err := s.reserve(n, pos+n); err != nil {
		return err
	}
	for i, id := range ids {
		metal.Slice[int32](s.ids)[i] = int32(id)
	}
	cos, sin := s.m.rope.Rotation(pos, n).Tables()
	copy(metal.Slice[float32](s.cos), cos)
	copy(metal.Slice[float32](s.sin), sin)
	eps := float32(c.RMSNormEps)
	scale := float32(1 / math.Sqrt(float64(c.HeadDim)))
	x, hb, y, qkv, ctx := s.x.At(0), s.h.At(0), s.y.At(0), s.qkv.At(0), s.ctx.At(0)
	linear := func(cb *metal.CommandBuffer, dst, src metal.Ref, w metal.Weights, out, in int) {
		k.Linear(cb, dst, out, src, in, n, w)
	}
	view := func() *tensor.Tensor { return tensor.FromData(metal.Slice[float32](s.x)[:n*h], n, h) }

	cb := g.gpu.Device.NewCommandBuffer()
	k.Embed(cb, x, g.embed, g.embedType, s.ids.At(0), c.VocabSize, n, h)
	for i, l := range g.layers {
		if keep != nil {
			if err := cb.Wait(); err != nil {
				return err
			}
			keep(view())
			cb = g.gpu.Device.NewCommandBuffer()
		}
		k.RMSNorm(cb, hb, x, l.attnNorm, h, n, 1, h, eps)
		linear(cb, qkv, hb, l.qkv, width, h)
		if l.qNorm.Buf != nil {
			k.RMSNorm(cb, qkv, qkv, l.qNorm, width, n, c.NumAttentionHeads, c.HeadDim, eps)
			kq := s.qkv.At(4 * qDim)
			k.RMSNorm(cb, kq, kq, l.kNorm, width, n, c.NumKeyValueHeads, c.HeadDim, eps)
		}
		k.RoPE(cb, qkv, s.cos.At(0), s.sin.At(0), width, n, c.NumAttentionHeads+c.NumKeyValueHeads, c.HeadDim)
		k.KVAppend(cb, s.caches[i], s.qkv.At(4*qDim), s.qkv.At(4*(qDim+kvDim)), width, n, pos)
		k.Attend(cb, ctx, qDim, qkv, width, s.caches[i], n, metal.AttendArgs{Heads: c.NumAttentionHeads, Scale: scale, Causal: true, Offset: pos})
		linear(cb, y, ctx, l.out, h, qDim)
		k.Add(cb, x, y, n*h)

		k.RMSNorm(cb, hb, x, l.ffNorm, h, n, 1, h, eps)
		linear(cb, s.gateUp.At(0), hb, l.gateUp, 2*ff, h)
		k.SwiGLU(cb, s.act.At(0), s.gateUp.At(0), n, ff)
		linear(cb, y, s.act.At(0), l.down, h, ff)
		k.Add(cb, x, y, n*h)
	}
	k.RMSNorm(cb, x, x, g.norm, h, n, 1, h, eps)
	if logits {
		k.Linear(cb, s.logits.At(0), c.VocabSize, s.x.At(4*(n-1)*h), h, 1, g.lmHead)
	}
	if err := cb.Wait(); err != nil {
		return err
	}
	if keep != nil {
		keep(view())
	}
	return nil
}

// logitsInto sets out [rows, vocab] to the output layer on y [rows, hidden],
// each row with the one-row kernel, so exactly as generation computes the
// logits of a hidden state.
func (g *gpuModel) logitsInto(c Config, out, y *tensor.Tensor) error {
	dev, k := g.gpu.Device, g.gpu.Kernels
	rows := y.Rows()
	in, err := dev.NewBuffer(4 * len(y.Data))
	if err != nil {
		return err
	}
	defer in.Release()
	res, err := dev.NewBuffer(4 * rows * c.VocabSize)
	if err != nil {
		return err
	}
	defer res.Release()
	copy(metal.Slice[float32](in), y.Data)
	cb := dev.NewCommandBuffer()
	for r := range rows {
		k.Linear(cb, res.At(4*r*c.VocabSize), c.VocabSize, in.At(4*r*c.HiddenSize), c.HiddenSize, 1, g.lmHead)
	}
	if err := cb.Wait(); err != nil {
		return err
	}
	copy(out.Data, metal.Slice[float32](res))
	return nil
}
