package inkling2

import (
	"math"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// state is one sequence being run: how many positions have run, and every
// layer visit's caches. The core's layers have one set per pass.
type state struct {
	m      *Model
	pos    int
	visits []visitState
}

// visitState is one layer visit's cache: its keys and values, and the last
// kernel−1 inputs of each of its four short convolutions, oldest first.
type visitState struct {
	keys, values  []float32 // [positions from start, kv], after their norms
	start         int       // the position of the first cached key: 0, or later in a sliding layer
	kIn, vIn      []float32 // the key and value projections before their convolutions
	attnIn, mlpIn []float32 // the inputs of the convolutions after attention and after the experts
}

func (m *Model) newState() *state {
	c := m.Config
	st := &state{m: m, visits: make([]visitState, c.Visits())}
	v := 0
	visit := func(i int) {
		_, kvHeads, hd := c.heads(i)
		k := c.SconvKernelSize - 1
		st.visits[v] = visitState{
			kIn: make([]float32, k*kvHeads*hd), vIn: make([]float32, k*kvHeads*hd),
			attnIn: make([]float32, k*c.HiddenSize), mlpIn: make([]float32, k*c.HiddenSize),
		}
		v++
	}
	m.eachVisit(visit)
	return st
}

// eachVisit calls fn with the layer of every visit, in the order they run.
func (m *Model) eachVisit(fn func(layer int)) {
	c := m.Config
	for i := range c.Loop.CoreStart {
		fn(i)
	}
	for range c.Loop.NumPasses {
		for i := c.Loop.CoreStart; i < c.Loop.CoreEnd; i++ {
			fn(i)
		}
	}
	for i := c.Loop.CoreEnd; i < c.NumHiddenLayers; i++ {
		fn(i)
	}
}

// forward runs ids at the next positions and returns the final normalized
// hidden states [len(ids), hidden]. keep, if not nil, sees the input to every
// layer visit and then the final output.
func (st *state) forward(ids []int, keep func(*tensor.Tensor)) *tensor.Tensor {
	m, c := st.m, st.m.Config
	n, h := len(ids), c.HiddenSize
	x := tensor.New(n, h)
	for t, id := range ids {
		row := x.Row(t)
		bf16Row(row, m.embed, id)
		rmsNorm(row, row, m.embedNorm, c.RMSNormEps)
	}

	var blocks []*tensor.Tensor // block states: inputs of the block-starting layers so far
	visit := 0
	run := func(i int, x *tensor.Tensor, messages []*tensor.Tensor) *tensor.Tensor {
		if keep != nil {
			keep(x)
		}
		y := st.layer(&m.layers[i], &st.visits[visit], x, blocks, messages)
		if c.boundary(i) {
			blocks = append(slices.Clip(blocks), x)
		}
		visit++
		return y
	}

	for i := range c.Loop.CoreStart {
		x = run(i, x, nil)
	}
	preX, preBlocks := x, slices.Clip(blocks)
	for i := c.Loop.CoreStart; i < c.Loop.CoreEnd; i++ {
		x = run(i, x, nil)
	}
	for range c.Loop.NumPasses - 1 {
		// The previous pass's block states, past the one every pass shares
		// (the core's input), and its output are what messages read.
		bank := append(slices.Clone(blocks[len(preBlocks)+1:]), x)
		x, blocks = preX, preBlocks
		for j := range c.loopLayers() {
			messages := st.messages(j, bank)
			x = run(c.Loop.CoreStart+j, x, messages)
		}
	}
	for i := c.Loop.CoreEnd; i < c.NumHiddenLayers; i++ {
		x = run(i, x, nil)
	}

	out := tensor.New(n, h)
	attnRes([]*tensor.Tensor{out}, m.finalQuery, blocks, x)
	for t := range n {
		rmsNorm(out.Row(t), out.Row(t), m.norm, c.RMSNormEps)
	}
	st.pos += n
	if keep != nil {
		keep(out)
	}
	return out
}

// messages returns core layer j's four messages [n, hidden] (for its query,
// key, value and expert inputs): attention of its learned queries over the
// previous pass's bank of states.
func (st *state) messages(j int, bank []*tensor.Tensor) []*tensor.Tensor {
	n, h := bank[0].Rows(), st.m.Config.HiddenSize
	out := make([]*tensor.Tensor, 4)
	for i := range out {
		out[i] = tensor.New(n, h)
	}
	attnRes(out, st.m.messageQueries[j], bank[:len(bank)-1], bank[len(bank)-1])
	return out
}

// layer runs one layer visit on x [n, hidden], with the block states so far
// and, in core passes after the first, its four messages, and returns the
// layer's output.
func (st *state) layer(ly *layer, vs *visitState, x *tensor.Tensor, blocks []*tensor.Tensor, messages []*tensor.Tensor) *tensor.Tensor {
	c := st.m.Config
	n, h := x.Rows(), c.HiddenSize
	eps := c.RMSNormEps

	// The attention inputs: queries, keys and values.
	var qIn, kIn, vIn *tensor.Tensor
	if ly.sliding {
		a := x
		if ly.attnQueries != nil {
			a = tensor.New(n, h)
			attnRes([]*tensor.Tensor{a}, ly.attnQueries, blocks, x)
		}
		if messages == nil {
			qIn = normRows(a, ly.inNorm, eps)
			kIn, vIn = qIn, qIn
		} else {
			qIn = normRows(st.m.gate(&ly.gates[0], a, messages[0]), ly.inNorm, eps)
			kIn = normRows(st.m.gate(&ly.gates[1], a, messages[1]), ly.inNorm, eps)
			vIn = normRows(st.m.gate(&ly.gates[2], a, messages[2]), ly.inNorm, eps)
		}
	} else {
		qa, ka, va := x, x, x
		if ly.attnQueries != nil {
			qa, ka, va = tensor.New(n, h), tensor.New(n, h), tensor.New(n, h)
			attnRes([]*tensor.Tensor{qa, ka, va}, ly.attnQueries, blocks, x)
		}
		if messages != nil {
			qa = st.m.gate(&ly.gates[0], qa, messages[0])
			ka = st.m.gate(&ly.gates[1], ka, messages[1])
			va = st.m.gate(&ly.gates[2], va, messages[2])
		}
		qIn, kIn, vIn = normRows(qa, ly.qNorm, eps), normRows(ka, ly.kNorm, eps), normRows(va, ly.vNorm, eps)
	}

	y := st.attention(ly, vs, qIn, kIn, vIn)
	y = staticConv(y, ly.postAttnConv, vs.attnIn, c.SconvKernelSize)

	// A block-starting layer's input becomes a block state, and its partial
	// sum restarts from the attention output.
	residual := y
	if c.boundary(ly.index) {
		blocks = append(slices.Clip(blocks), x)
	} else {
		ops.Add(residual.Data, x.Data)
	}

	mIn := tensor.New(n, h)
	attnRes([]*tensor.Tensor{mIn}, ly.mlpQuery, blocks, residual)
	if messages != nil {
		mIn = st.m.gate(&ly.gates[3], mIn, messages[3])
	}
	mOut := st.m.experts(ly, normRows(mIn, ly.mlpNorm, eps))
	mOut = staticConv(mOut, ly.postMLPConv, vs.mlpIn, c.SconvKernelSize)
	ops.Add(residual.Data, mOut.Data)
	return residual
}

// gate returns the message gate's mix of x and message, both [n, hidden].
func (m *Model) gate(g *messageGate, x, message *tensor.Tensor) *tensor.Tensor {
	c := m.Config
	n, h, r := x.Rows(), c.HiddenSize, c.Loop.GateRank
	pair := tensor.New(n, 2*h)
	for t := range n {
		rmsNorm(pair.Row(t)[:h], x.Row(t), nil, c.RMSNormEps)
		rmsNorm(pair.Row(t)[h:], message.Row(t), nil, c.RMSNormEps)
	}
	down := tensor.New(n, r)
	ops.LinearInto(down, pair, g.down, nil)
	forget, gain := tensor.New(n, h), tensor.New(n, h)
	ops.LinearInto(forget, down, g.forget, nil)
	ops.LinearInto(gain, down, g.gain, nil)
	out := tensor.New(n, h)
	for i, v := range x.Data {
		f := float32(math.Tanh(float64(forget.Data[i])))
		gn := float32(math.Tanh(float64(gain.Data[i])))
		out.Data[i] = (1-f)*v + gn*message.Data[i]
	}
	return out
}

// attnRes sets outs[q] [n, hidden] to the attention of query q (queries
// holds len(outs) of them) over the RMS-normalized block states and partial
// sum at each position: transformers' Inkling2LoopBlockAttentionResidual,
// whose keys and values are both the normalized states.
func attnRes(outs []*tensor.Tensor, queries []float32, blocks []*tensor.Tensor, partial *tensor.Tensor) {
	n, h := partial.Rows(), partial.Cols()
	scale := float32(1 / math.Sqrt(float64(h)))
	sources := append(slices.Clone(blocks), partial)
	parallel(n, func(t int) {
		normed := make([]float32, len(sources)*h)
		for j, s := range sources {
			rmsNorm(normed[j*h:(j+1)*h], s.Row(t), nil, 1e-6)
		}
		p := make([]float32, len(sources))
		for qi, out := range outs {
			q := queries[qi*h : (qi+1)*h]
			for j := range sources {
				p[j] = dot(q, normed[j*h:(j+1)*h]) * scale
			}
			softmax(p)
			o := out.Row(t)
			clear(o)
			for j, w := range p {
				for k, v := range normed[j*h : (j+1)*h] {
					o[k] += w * v
				}
			}
		}
	})
}

// attention returns attention layer ly's output [n, hidden] for its query,
// key and value inputs, appending this visit's keys and values to its cache.
func (st *state) attention(ly *layer, vs *visitState, qIn, kIn, vIn *tensor.Tensor) *tensor.Tensor {
	c := st.m.Config
	n, h := qIn.Rows(), c.HiddenSize
	heads, kvHeads, hd, dRel := ly.heads, ly.kvHeads, ly.headDim, c.DRel
	qDim, kv, kernel := heads*hd, kvHeads*hd, c.SconvKernelSize
	gateOff, relOff, sinkOff := qDim, 2*qDim, 2*qDim+heads*dRel

	qp := tensor.New(n, sinkOff+heads)
	ops.LinearInto(qp, qIn, ly.q, nil)
	kp, vp := tensor.New(n, ly.k.N()), tensor.New(n, ly.v.N())
	ops.LinearInto(kp, kIn, ly.k, nil)
	ops.LinearInto(vp, vIn, ly.v, nil)

	var keys, values *tensor.Tensor
	if ly.sliding {
		keys = depthwiseConv(kp, kv, ly.kConv, vs.kIn, kernel)
		values = depthwiseConv(vp, kv, ly.vConv, vs.vIn, kernel)
	} else {
		keys = dynamicConv(kp, kv, kvHeads, ly.kConv, vs.kIn, kernel)
		values = dynamicConv(vp, kv, kvHeads, ly.vConv, vs.vIn, kernel)
	}

	qkClip, relClip := float32(c.QKL2ClipThreshold), float32(c.RelL2ClipThreshold)
	for t := range n {
		var tau float32 = 1
		if !ly.sliding {
			tau = c.logScale(st.pos + t)
		}
		row := qp.Row(t)
		for hh := range heads {
			q := row[hh*hd : (hh+1)*hd]
			scaledClip(q, ly.qScale[hh*hd:(hh+1)*hd], qkClip, tau)
			r := row[relOff+hh*dRel : relOff+(hh+1)*dRel]
			scaledClip(r, ly.relScale[hh*dRel:(hh+1)*dRel], relClip, tau)
		}
		k, v := keys.Row(t), values.Row(t)
		for hh := range kvHeads {
			scaledClip(k[hh*hd:(hh+1)*hd], nil, qkClip, 1)
			rmsNorm(v[hh*hd:(hh+1)*hd], v[hh*hd:(hh+1)*hd], nil, c.RMSNormEps)
		}
	}
	vs.keys = append(vs.keys, keys.Data...)
	vs.values = append(vs.values, values.Data...)

	ctx := tensor.New(n, qDim)
	group := heads / kvHeads
	scale := float32(1 / math.Sqrt(float64(hd)))
	ext := ly.relExtent
	parallel(n*heads, func(task int) {
		t, hh := task/heads, task%heads
		p := st.pos + t // this query's position
		lo := 0
		if ly.sliding {
			lo = max(0, p-c.SlidingWindowSize+1)
		}
		row := qp.Row(t)
		q := row[hh*hd : (hh+1)*hd]
		rel := row[relOff+hh*dRel : relOff+(hh+1)*dRel]
		kh := hh / group
		scores := make([]float32, p-lo+2) // keys lo..p, then the sink
		for j := lo; j <= p; j++ {
			at := (j - vs.start) * kv
			s := dot(q, vs.keys[at+kh*hd:at+(kh+1)*hd]) * scale
			if d := p - j; d < ext {
				var b float32
				for r, w := range rel {
					b += w * ly.relProj[r*ext+d]
				}
				s += b
			}
			scores[j-lo] = s
		}
		scores[p-lo+1] = row[sinkOff+hh]
		softmax(scores)
		o := ctx.Row(t)[hh*hd : (hh+1)*hd]
		for j := lo; j <= p; j++ {
			w := scores[j-lo]
			at := (j - vs.start) * kv
			for i, v := range vs.values[at+kh*hd : at+(kh+1)*hd] {
				o[i] += w * v
			}
		}
	})

	// A sliding layer's later queries see only the last window−1 positions
	// before them. Keep those, trimming once twice the window has built up.
	if keep := c.SlidingWindowSize - 1; ly.sliding && st.pos+n-vs.start > 2*c.SlidingWindowSize {
		drop := st.pos + n - keep - vs.start
		vs.keys = append(vs.keys[:0], vs.keys[drop*kv:]...)
		vs.values = append(vs.values[:0], vs.values[drop*kv:]...)
		vs.start += drop
	}

	// The sigmoid output gate, from the query input, then the projection.
	for t := range n {
		gate := qp.Row(t)[gateOff : gateOff+qDim]
		o := ctx.Row(t)
		for i, g := range gate {
			o[i] *= sigmoid(g)
		}
	}
	y := tensor.New(n, h)
	ops.LinearInto(y, ctx, ly.out, nil)
	return y
}

// scaledClip multiplies x by scale (if not nil), scales it down to norm
// limit when longer, then multiplies it by tau: transformers' L2 clips,
// (x·scale) · min(limit/(‖x·scale‖+1e-6), 1), and log scale.
func scaledClip(x, scale []float32, limit, tau float32) {
	if scale != nil {
		for i := range x {
			x[i] *= scale[i]
		}
	}
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	f := min(limit/(float32(math.Sqrt(sum))+1e-6), 1)
	for i := range x {
		x[i] *= f
		if tau != 1 {
			x[i] *= tau
		}
	}
}

// depthwiseConv returns the short causal convolution of the first channels
// columns of x [n, ≥channels], with weights w [channels, kernel] and the
// last kernel−1 inputs before x in past (updated), plus its input, over
// √(kernel+1): transformers' Inkling2LoopShortConvolution.
func depthwiseConv(x *tensor.Tensor, channels int, w, past []float32, kernel int) *tensor.Tensor {
	n := x.Rows()
	in := convInputs(x, channels, past, kernel)
	out := tensor.New(n, channels)
	norm := float32(math.Sqrt(float64(kernel + 1)))
	parallel(n, func(t int) {
		o := out.Row(t)
		for j := range kernel {
			for ch, v := range in(t - (kernel - 1) + j) {
				o[ch] += w[ch*kernel+j] * v
			}
		}
		for ch, v := range in(t) {
			o[ch] = (o[ch] + v) / norm
		}
	})
	keepPast(in, n, channels, past, kernel)
	return out
}

// dynamicConv is the full-attention layers' convolution: each tap is a
// static weight per channel, static [kernel, channels] (tap 0 the current
// position), plus a weight per head and position, from the kernel·heads
// columns of x after the first channels. Its output adds its input and
// divides by √(2·kernel+1): transformers' Inkling2LoopDynamicShortConvolution.
func dynamicConv(x *tensor.Tensor, channels, heads int, static, past []float32, kernel int) *tensor.Tensor {
	n := x.Rows()
	in := convInputs(x, channels, past, kernel)
	out := tensor.New(n, channels)
	hd := channels / heads
	norm := float32(math.Sqrt(float64(2*kernel + 1)))
	parallel(n, func(t int) {
		o := out.Row(t)
		dyn := x.Row(t)[channels:]
		cur := in(t)
		for ch := range channels {
			var sum float32
			for lag := range kernel {
				w := static[lag*channels+ch] + dyn[(ch/hd)*kernel+lag]
				sum += in(t - lag)[ch] * w
			}
			o[ch] = (sum + cur[ch]) / norm
		}
	})
	keepPast(in, n, channels, past, kernel)
	return out
}

// convInputs returns the convolution input at position u of this pass:
// x's row u, or for negative u the saved past inputs.
func convInputs(x *tensor.Tensor, channels int, past []float32, kernel int) func(u int) []float32 {
	return func(u int) []float32 {
		if u >= 0 {
			return x.Row(u)[:channels]
		}
		return past[(kernel-1+u)*channels : (kernel+u)*channels]
	}
}

// keepPast saves the last kernel−1 inputs, some of which may still be in
// past, as the next pass's past.
func keepPast(in func(int) []float32, n, channels int, past []float32, kernel int) {
	keep := make([]float32, len(past))
	for j := range kernel - 1 {
		copy(keep[j*channels:(j+1)*channels], in(n-(kernel-1)+j))
	}
	copy(past, keep)
}

// staticConv is depthwiseConv over all of x, with weights w [channels,
// kernel].
func staticConv(x *tensor.Tensor, w, past []float32, kernel int) *tensor.Tensor {
	return depthwiseConv(x, x.Cols(), w, past, kernel)
}

// experts returns the routed experts' output for x [n, hidden], each
// position's experts weighted and summed in ascending expert order, as
// transformers accumulates them.
func (m *Model) experts(ly *layer, x *tensor.Tensor) *tensor.Tensor {
	c := m.Config
	n, h, ne, topK, ff := x.Rows(), c.HiddenSize, c.NRoutedExperts, c.NumExpertsPerTok, c.IntermediateSize

	logits := tensor.New(n, ne)
	ops.LinearInto(logits, x, ly.router, nil)
	type pick struct {
		expert int
		weight float32
	}
	picks := make([][]pick, n)
	byExpert := make([][]int, ne) // positions routed to each expert
	for t := range n {
		scores := logits.Row(t)
		for e := range scores {
			scores[e] = sigmoid(scores[e])
		}
		order := make([]int, ne)
		for e := range order {
			order[e] = e
		}
		slices.SortStableFunc(order, func(a, b int) int {
			sa, sb := scores[a]+ly.routerBias[a], scores[b]+ly.routerBias[b]
			switch {
			case sa > sb:
				return -1
			case sa < sb:
				return 1
			}
			return 0
		})
		chosen := slices.Clone(order[:topK])
		slices.Sort(chosen)
		var sum float32
		for _, e := range chosen {
			sum += scores[e]
		}
		for _, e := range chosen {
			picks[t] = append(picks[t], pick{e, scores[e] / (sum + 1e-20) * float32(c.RouteScale)})
			byExpert[e] = append(byExpert[e], t)
		}
	}

	// outs[t][slot] is position t's slot-th expert's output.
	outs := make([][]float32, n)
	for t := range outs {
		outs[t] = make([]float32, topK*h)
	}
	gateCap, upCap := float32(c.MLPGateSoftcap), float32(c.MLPUpSoftcap)
	for e, rows := range byExpert {
		if len(rows) == 0 {
			continue
		}
		xs := tensor.New(len(rows), h)
		for i, t := range rows {
			copy(xs.Row(i), x.Row(t))
		}
		gu := linearMicro(xs, ly.gateUp[e])
		act := tensor.New(len(rows), ff)
		for i := range len(rows) {
			g, a := gu.Row(i), act.Row(i)
			for j := range ff {
				gate, up := g[2*j], g[2*j+1]
				a[j] = softcap(gate, gateCap) * sigmoid(gate) * softcap(up, upCap)
			}
		}
		d := linearMicro(act, ly.down[e])
		for i, t := range rows {
			slot := slices.IndexFunc(picks[t], func(p pick) bool { return p.expert == e })
			w := picks[t][slot].weight
			o := outs[t][slot*h : (slot+1)*h]
			for j, v := range d.Row(i) {
				o[j] = v * w
			}
		}
	}
	y := tensor.New(n, h)
	for t := range n {
		row := y.Row(t)
		for slot := range topK {
			for j, v := range outs[t][slot*h : (slot+1)*h] {
				row[j] += v
			}
		}
	}
	return y
}

// denseRows is the number of rows from which a product over microscaled
// weights dequantizes them and runs the dense product instead: the
// dequantization and packing cost about as much as 28 rows decoded as they
// go (an expert's gate and up projections, on an M2 Max).
var denseRows = 32

// linearMicro returns x [n, cols] times wᵀ, [n, rows].
func linearMicro(x *tensor.Tensor, w *ops.Microscaled) *tensor.Tensor {
	n := x.Rows()
	if n >= denseRows {
		dense := tensor.New(w.Rows(), w.Cols())
		w.Dequantize(dense.Data, 0)
		return ops.MatMulT(x, dense)
	}
	y := tensor.New(n, w.Rows())
	ops.MicroLinearInto(y.Data, w.Rows(), x.Data, w.Cols(), n, w)
	return y
}

// normRows returns RMSNorm of each row of x with weight w.
func normRows(x *tensor.Tensor, w []float32, eps float64) *tensor.Tensor {
	out := tensor.New(x.Rows(), x.Cols())
	for t := range x.Rows() {
		rmsNorm(out.Row(t), x.Row(t), w, eps)
	}
	return out
}

// rmsNorm sets dst to src scaled to unit root mean square, then by w if not
// nil.
func rmsNorm(dst, src, w []float32, eps float64) {
	var sum float64
	for _, v := range src {
		sum += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(sum/float64(len(src))+eps)
	for j, v := range src {
		r := float32(float64(v) * inv)
		if w != nil {
			r *= w[j]
		}
		dst[j] = r
	}
}

// softmax replaces x with softmax(x), computed in float64.
func softmax(x []float32) {
	hi := float32(math.Inf(-1))
	for _, v := range x {
		hi = max(hi, v)
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - hi))
		x[i] = float32(e)
		sum += e
	}
	for i := range x {
		x[i] = float32(float64(x[i]) / sum)
	}
}

func dot(a, b []float32) float32 {
	var s float32
	for i, v := range a {
		s += v * b[i]
	}
	return s
}

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// softcap is tanh(x/c)·c.
func softcap(x, c float32) float32 { return float32(math.Tanh(float64(x/c))) * c }

func softcapInPlace(x []float32, c float32) {
	for i, v := range x {
		x[i] = softcap(v, c)
	}
}

// parallel runs fn(0) … fn(n−1) on up to GOMAXPROCS goroutines.
func parallel(n int, fn func(int)) {
	workers := min(runtime.GOMAXPROCS(0), n)
	if workers <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			for i := int(next.Add(1) - 1); i < n; i = int(next.Add(1) - 1) {
				fn(i)
			}
		})
	}
	wg.Wait()
}
