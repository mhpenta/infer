package qwen35

import (
	"math"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// state is one sequence being run: how many positions have run, the
// full-attention layers' key/value caches, and the linear layers' recurrent
// state, the last kernel−1 convolution inputs and the delta rule's matrices.
type state struct {
	m      *Model
	pos    int
	caches []*ops.KVCache // per layer; nil for linear layers
	conv   [][]float32    // per layer: [kernel−1, convDim], oldest first; nil for full layers
	delta  [][]float32    // per layer: the delta rule's state; nil for full layers
}

// newState returns an empty sequence with cache room for capacity positions.
func (m *Model) newState(capacity int) *state {
	c := m.Config
	st := &state{m: m}
	for _, l := range m.layers {
		if l.full {
			st.caches = append(st.caches, ops.NewKVCache(c.NumKeyValueHeads, c.HeadDim, capacity))
			st.conv = append(st.conv, nil)
			st.delta = append(st.delta, nil)
			continue
		}
		st.caches = append(st.caches, nil)
		st.conv = append(st.conv, make([]float32, (c.LinearConvKernelDim-1)*c.convDim()))
		st.delta = append(st.delta, make([]float32, c.deltaArgs().StateSize()))
	}
	return st
}

// snapshot is a copy of a state's position and recurrent state. The key/value
// caches are not copied: restoring truncates them.
type snapshot struct {
	pos         int
	conv, delta [][]float32
}

func (st *state) save() snapshot {
	s := snapshot{pos: st.pos}
	for i := range st.conv {
		s.conv = append(s.conv, clone(st.conv[i]))
		s.delta = append(s.delta, clone(st.delta[i]))
	}
	return s
}

// restore returns the state to s, which an earlier save of it took.
func (st *state) restore(s snapshot) {
	st.pos = s.pos
	for i, c := range st.caches {
		if c != nil {
			c.Truncate(s.pos)
		}
		copy(st.conv[i], s.conv[i])
		copy(st.delta[i], s.delta[i])
	}
}

func clone(x []float32) []float32 {
	if x == nil {
		return nil
	}
	return append([]float32(nil), x...)
}

// forward runs ids at the next positions and returns the final normalized
// hidden states [len(ids), hidden]. keep, if not nil, sees the input to every
// layer and then the final output.
func (st *state) forward(ids []int, keep func(*tensor.Tensor)) *tensor.Tensor {
	m, c := st.m, st.m.Config
	n := len(ids)
	x := tensor.New(n, c.HiddenSize)
	for i, id := range ids {
		m.embedRow(id, x.Row(i))
	}
	h, y := tensor.New(n, c.HiddenSize), tensor.New(n, c.HiddenSize)
	gateUp, act := tensor.New(n, 2*c.IntermediateSize), tensor.New(n, c.IntermediateSize)
	var full fullBuffers
	var linear linearBuffers
	for i, l := range m.layers {
		if keep != nil {
			keep(x)
		}
		rmsNormInto(h, x, l.attnNorm, c.RMSNormEps)
		if l.full {
			st.attention(i, l, h, y, &full)
		} else {
			st.linearAttention(i, l, h, y, &linear)
		}
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

// fullBuffers are a forward pass's scratch tensors for full attention,
// allocated by the first such layer.
type fullBuffers struct {
	qgkv, ctx *tensor.Tensor
	rot       *ops.Rotation
}

// attention sets y to full-attention layer i's output for h, appending the
// positions' keys and values to its cache.
func (st *state) attention(i int, l layer, h, y *tensor.Tensor, b *fullBuffers) {
	c := st.m.Config
	n, qDim, kvDim, hd := h.Rows(), c.qDim(), c.kvDim(), c.HeadDim
	width := 2*qDim + 2*kvDim // q, gate, k, v
	if b.qgkv == nil {
		b.qgkv, b.ctx = tensor.New(n, width), tensor.New(n, qDim)
		b.rot = st.m.rope.Rotation(st.pos, n)
	}
	qgkv, ctx := b.qgkv, b.ctx
	ops.LinearInto(qgkv, h, l.qgkv, nil)
	kOff, vOff := 2*qDim, 2*qDim+kvDim
	for r := range n {
		row := qgkv.Row(r)
		ops.RMSNorm(tensor.FromData(row[:qDim], c.NumAttentionHeads, hd), l.qNorm, c.RMSNormEps)
		ops.RMSNorm(tensor.FromData(row[kOff:vOff], c.NumKeyValueHeads, hd), l.kNorm, c.RMSNormEps)
	}
	// Only each head's first rotaryDim dimensions rotate.
	for head := range c.NumAttentionHeads {
		b.rot.Apply(qgkv.Data[head*hd:], width, 1)
	}
	for head := range c.NumKeyValueHeads {
		b.rot.Apply(qgkv.Data[kOff+head*hd:], width, 1)
	}
	cache := st.caches[i]
	cache.Append(qgkv.Data[kOff:], width, qgkv.Data[vOff:], width, n)
	cache.Attend(ctx.Data, qDim, qgkv.Data, width, n, ops.AttentionArgs{
		Heads: c.NumAttentionHeads, KVHeads: c.NumKeyValueHeads,
		Scale:  float32(1 / math.Sqrt(float64(hd))),
		Causal: true, Offset: st.pos,
	})
	for r := range n {
		gate := qgkv.Row(r)[qDim : 2*qDim]
		for j, g := range gate {
			ctx.Data[r*qDim+j] *= sigmoid(g)
		}
	}
	ops.LinearInto(y, ctx, l.out, nil)
}

// linearBuffers are a forward pass's scratch tensors for linear attention,
// allocated by the first such layer.
type linearBuffers struct {
	proj, mixed, core *tensor.Tensor
	decay, beta       []float32
}

// linearAttention sets y to linear layer i's output for h, advancing its
// recurrent state.
func (st *state) linearAttention(i int, l layer, h, y *tensor.Tensor, b *linearBuffers) {
	c := st.m.Config
	n, cd, kd, vd := h.Rows(), c.convDim(), c.keyDim(), c.valueDim()
	nk, nv, dk, dv := c.LinearNumKeyHeads, c.LinearNumValueHeads, c.LinearKeyHeadDim, c.LinearValueHeadDim
	width := cd + vd + 2*nv // the convolution's input, z, b, a
	if b.proj == nil {
		b.proj, b.mixed, b.core = tensor.New(n, width), tensor.New(n, cd), tensor.New(n, vd)
		b.decay, b.beta = make([]float32, n*nv), make([]float32, n*nv)
	}
	proj, mixed, core := b.proj, b.mixed, b.core
	ops.LinearInto(proj, h, l.in, nil)
	st.convolve(i, l, proj, mixed)

	// L2-normalize each head's queries and keys, then scale the queries.
	scale := float32(1 / math.Sqrt(float64(dk)))
	for r := range n {
		row := mixed.Row(r)
		for head := range nk {
			q, k := row[head*dk:(head+1)*dk], row[kd+head*dk:kd+(head+1)*dk]
			l2Normalize(q)
			for j := range q {
				q[j] *= scale
			}
			l2Normalize(k)
		}
	}
	for r := range n {
		row := proj.Row(r)
		bs, as := row[cd+vd:cd+vd+nv], row[cd+vd+nv:]
		for head := range nv {
			b.beta[r*nv+head] = sigmoid(bs[head])
			g := l.negA[head] * softplus(as[head]+l.dtBias[head])
			b.decay[r*nv+head] = float32(math.Exp(float64(g)))
		}
	}
	ops.GatedDeltaRule(core.Data, vd, mixed.Data, cd, mixed.Data[kd:], cd, mixed.Data[2*kd:], cd, b.decay, b.beta, st.delta[i], n, c.deltaArgs())

	// Each head's output: RMSNorm, scaled by the norm's weight and SiLU(z).
	for r := range n {
		o, z := core.Row(r), proj.Row(r)[cd:cd+vd]
		for head := range nv {
			gatedRMSNorm(o[head*dv:(head+1)*dv], z[head*dv:(head+1)*dv], l.headNorm.Data, c.RMSNormEps)
		}
	}
	ops.LinearInto(y, core, l.linearOut, nil)
}

// convolve sets mixed [n, convDim] to SiLU of linear layer i's depthwise
// causal convolution over the first convDim columns of proj, continuing from
// the inputs its state keeps, and updates that state.
func (st *state) convolve(i int, l layer, proj, mixed *tensor.Tensor) {
	c := st.m.Config
	n, cd, kernel := proj.Rows(), c.convDim(), c.LinearConvKernelDim
	past := st.conv[i]               // [kernel−1, cd], oldest first
	input := func(u int) []float32 { // position u of this pass; negative reaches into past
		if u >= 0 {
			return proj.Row(u)[:cd]
		}
		return past[(kernel-1+u)*cd : (kernel+u)*cd]
	}
	for t := range n {
		out := mixed.Row(t)
		clear(out)
		for j := range kernel {
			in := input(t - (kernel - 1) + j)
			for ch, v := range in {
				out[ch] += l.conv[ch*kernel+j] * v
			}
		}
		ops.SiLU(out)
	}
	// Keep the last kernel−1 inputs, which may still be in past.
	keep := make([]float32, len(past))
	for j := range kernel - 1 {
		copy(keep[j*cd:(j+1)*cd], input(n-(kernel-1)+j))
	}
	copy(past, keep)
}

// rmsNormInto sets dst to RMSNorm(src) with scale w.
func rmsNormInto(dst, src, w *tensor.Tensor, eps float64) {
	copy(dst.Data, src.Data)
	ops.RMSNorm(dst, w, eps)
}

// gatedRMSNorm sets o to RMSNorm(o)·w·SiLU(z), Qwen3.5's gated head norm.
func gatedRMSNorm(o, z, w []float32, eps float64) {
	var sum float64
	for _, v := range o {
		sum += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(sum/float64(len(o))+eps)
	for j, v := range o {
		g := z[j]
		o[j] = w[j] * float32(float64(v)*inv) * (g / (1 + float32(math.Exp(-float64(g)))))
	}
}

// l2Normalize scales x to unit length, as transformers' l2norm does with its
// epsilon of 1e-6 under the root.
func l2Normalize(x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(sum+1e-6)
	for j, v := range x {
		x[j] = float32(float64(v) * inv)
	}
}

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// softplus is log(1+eˣ), and x itself above 20, as torch's default.
func softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}
