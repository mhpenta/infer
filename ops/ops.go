// Package ops implements the inference kernels: linear layers, normalization,
// activations, softmax and attention.
//
// Matrix products go through gemm (matmul.go), which uses a SIMD microkernel
// where one exists (AVX2 on amd64, NEON on arm64) and portable Go elsewhere or
// with the purego build tag. Shape mismatches panic, because they are
// programming errors rather than bad input.
package ops

import (
	"fmt"
	"math"

	"github.com/mhpenta/infer/tensor"
)

// Linear computes x·wᵀ + b, where x is [n, in], w is an nn.Linear weight
// [out, in] packed with Pack, and b is [out] or nil. It returns a new [n, out]
// tensor.
func Linear(x *tensor.Tensor, w *Packed, b *tensor.Tensor) *tensor.Tensor {
	y := tensor.New(x.Rows(), w.N())
	LinearInto(y, x, w, b)
	return y
}

// LinearInto is Linear writing into y [n, out], whose prior contents do not
// matter.
func LinearInto(y, x *tensor.Tensor, w *Packed, b *tensor.Tensor) {
	if b != nil && b.Size() != w.N() {
		panic(fmt.Sprintf("ops.LinearInto: w has %d outputs but b is %v", w.N(), b.Shape))
	}
	if x.Cols() != w.K() || y.Rows() != x.Rows() || y.Cols() != w.N() {
		panic(fmt.Sprintf("ops.LinearInto: y is %v, x is %v, w is %dx%d", y.Shape, x.Shape, w.N(), w.K()))
	}
	var bias []float32
	if b != nil {
		bias = b.Data
	}
	gemm(x.Rows(), x.Data, w.K(), w, bias, y.Data, w.N(), workersFor(x.Rows()*w.N()*w.K()))
}

// Add adds src to dst element-wise, in place.
func Add(dst, src []float32) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("ops.Add: lengths %d and %d", len(dst), len(src)))
	}
	for i := range dst {
		dst[i] += src[i]
	}
}

// LayerNorm normalizes each row of x in place to zero mean and unit variance,
// then applies the per-feature weight and bias. Statistics are accumulated in
// float64.
func LayerNorm(x, weight, bias *tensor.Tensor, eps float64) {
	checkLayerNorm("LayerNorm", x, weight, bias)
	for i := range x.Rows() {
		row := x.Row(i)
		var sum float64
		for _, v := range row {
			sum += float64(v)
		}
		normalizeRow(row, sum, weight.Data, bias.Data, eps)
	}
}

// AddLayerNorm sets x to LayerNorm(x + residual), in place: a transformer
// sublayer's residual add fused into the normalization's first pass.
func AddLayerNorm(x, residual, weight, bias *tensor.Tensor, eps float64) {
	checkLayerNorm("AddLayerNorm", x, weight, bias)
	if residual.Rows() != x.Rows() || residual.Cols() != x.Cols() {
		panic(fmt.Sprintf("ops.AddLayerNorm: x is %v, residual %v", x.Shape, residual.Shape))
	}
	for i := range x.Rows() {
		row, res := x.Row(i), residual.Row(i)
		var sum float64
		for j, v := range res {
			row[j] += v
			sum += float64(row[j])
		}
		normalizeRow(row, sum, weight.Data, bias.Data, eps)
	}
}

func checkLayerNorm(name string, x, weight, bias *tensor.Tensor) {
	if cols := x.Cols(); weight.Size() != cols || bias.Size() != cols {
		panic(fmt.Sprintf("ops.%s: x is %v, weight %v, bias %v", name, x.Shape, weight.Shape, bias.Shape))
	}
}

// normalizeRow finishes LayerNorm on one row given the sum of its values.
func normalizeRow(row []float32, sum float64, weight, bias []float32, eps float64) {
	mean := sum / float64(len(row))
	var variance float64
	for _, v := range row {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= float64(len(row))
	inv := 1 / math.Sqrt(variance+eps)
	for j, v := range row {
		row[j] = float32((float64(v)-mean)*inv)*weight[j] + bias[j]
	}
}

// GELU applies the exact (erf-based) GELU activation in place, matching
// PyTorch's default nn.GELU and BERT's "gelu" to within about 1e-6.
func GELU(x []float32) {
	geluSlice(x)
}

// Softmax replaces x with softmax(x), in place.
func Softmax(x []float32) {
	softmaxScaled(x, 1)
}

// softmaxScaled replaces x with softmax(scale·x), in place, for scale > 0.
// Folding the scale into the exponent saves attention a pass over its scores:
// the maximum of x is also the maximum of scale·x.
func softmaxScaled(x []float32, scale float32) {
	if len(x) == 0 {
		return
	}
	peak := maxSlice(x)
	sum := expShiftScale(x, peak, scale)
	scaleSlice(x, float32(1/sum))
}

// Attention computes multi-head scaled dot-product attention without masking.
// q is [seq, hidden]; k and v are [kvSeq, hidden]; hidden is split evenly into
// heads. scores = q·kᵀ·scale per head, softmaxed over keys, then applied to v.
// It returns the [seq, hidden] context, with heads concatenated.
func Attention(q, k, v *tensor.Tensor, heads int, scale float32) *tensor.Tensor {
	out := tensor.New(q.Rows(), q.Cols())
	AttentionInto(out, q, k, v, heads, scale)
	return out
}

// AttentionInto is Attention writing into out [seq, hidden], whose prior
// contents do not matter.
func AttentionInto(out, q, k, v *tensor.Tensor, heads int, scale float32) {
	seq, hidden := q.Rows(), q.Cols()
	kvSeq := k.Rows()
	if k.Cols() != hidden || v.Cols() != hidden || v.Rows() != kvSeq || hidden%heads != 0 ||
		out.Rows() != seq || out.Cols() != hidden {
		panic(fmt.Sprintf("ops.AttentionInto: out %v, q %v, k %v, v %v, %d heads", out.Shape, q.Shape, k.Shape, v.Shape, heads))
	}
	AttentionStrided(out.Data, hidden, q.Data, hidden, k.Data, hidden, v.Data, hidden, seq, kvSeq, hidden, AttentionArgs{Heads: heads, Scale: scale})
}

// AttentionQKVInto is self-attention on qkv [seq, 3·hidden], the output of
// one Linear whose columns hold q, then k, then v. It writes the context into
// out [seq, hidden]; out's prior contents do not matter.
func AttentionQKVInto(out, qkv *tensor.Tensor, heads int, scale float32) {
	seq, hidden := out.Rows(), out.Cols()
	if qkv.Rows() != seq || qkv.Cols() != 3*hidden || hidden%heads != 0 {
		panic(fmt.Sprintf("ops.AttentionQKVInto: out %v, qkv %v, %d heads", out.Shape, qkv.Shape, heads))
	}
	d := qkv.Data
	AttentionStrided(out.Data, hidden, d, 3*hidden, d[hidden:], 3*hidden, d[2*hidden:], 3*hidden, seq, seq, hidden, AttentionArgs{Heads: heads, Scale: scale})
}

// AttentionArgs configures AttentionStrided.
type AttentionArgs struct {
	Heads int
	// KVHeads is the number of key and value heads, for grouped-query
	// attention: query head h reads key/value head h/(Heads/KVHeads). 0
	// means Heads.
	KVHeads int
	// Scale multiplies q·kᵀ: 1/√headDim for BERT, 1 for T5.
	Scale float32
	// Bias, if not nil, is added to the scaled scores before the softmax:
	// [Heads, seq, kvSeq], row-major (T5's relative position bias).
	Bias []float32
	// Causal hides from query i every key after position Offset+i: the
	// queries are positions Offset.. of the key sequence, as when decoding
	// with a cache of Offset earlier positions.
	Causal bool
	Offset int
}

// kvHeads returns the number of key and value heads.
func (a AttentionArgs) kvHeads() int {
	if a.KVHeads == 0 {
		return a.Heads
	}
	return a.KVHeads
}

// checkHeads panics unless the query heads split evenly into kvHeads groups.
func (a AttentionArgs) checkHeads(name string) {
	if kv := a.kvHeads(); a.Heads <= 0 || kv <= 0 || a.Heads%kv != 0 {
		panic(fmt.Sprintf("ops.%s: %d query heads do not split into %d key/value heads", name, a.Heads, kv))
	}
}

// AttentionStrided computes multi-head attention, softmax(Scale·q·kᵀ + Bias,
// with the causal mask)·v for each head, on strided views: row i of q starts
// at q[i*ldq], rows of k and v at multiples of ldk and ldv, and the context
// for query i is written to out[i*ldOut:]. q and out hold hidden columns,
// split evenly into args.Heads; k and v hold args.KVHeads heads of the same
// size. Each key/value head is packed once, then heads run in parallel, each
// head's two products serially on views of its columns.
func AttentionStrided(out []float32, ldOut int, q []float32, ldq int, k []float32, ldk int, v []float32, ldv int, seq, kvSeq, hidden int, args AttentionArgs) {
	args.checkHeads("AttentionStrided")
	heads, kvHeads := args.Heads, args.kvHeads()
	if hidden%heads != 0 {
		panic(fmt.Sprintf("ops.AttentionStrided: %d columns do not split into %d heads", hidden, heads))
	}
	if args.Bias != nil && len(args.Bias) != heads*seq*kvSeq {
		panic(fmt.Sprintf("ops.AttentionStrided: bias has %d values, want %d·%d·%d", len(args.Bias), heads, seq, kvSeq))
	}
	if args.Causal && args.Offset+seq > kvSeq {
		panic(fmt.Sprintf("ops.AttentionStrided: %d causal queries after %d cached positions need %d keys, have %d", seq, args.Offset, args.Offset+seq, kvSeq))
	}
	headDim := hidden / heads
	workers := workersFor(2 * seq * kvSeq * hidden)
	kSize, vSize := packedSize(kvSeq, headDim), packedSize(headDim, kvSeq)
	buf := getScratch(kvHeads * (kSize + vSize))
	defer scratch.Put(buf)
	kps, vps := make([]Packed, kvHeads), make([]Packed, kvHeads)
	forEach(kvHeads, workers, func(g int) {
		off, packed := g*headDim, (*buf)[g*(kSize+vSize):]
		kps[g] = packTInto(packed[:kSize], k[off:], kvSeq, headDim, ldk)
		vps[g] = packNInto(packed[kSize:kSize+vSize], v[off:], kvSeq, headDim, ldv)
	})
	attendHeads(out, ldOut, q, ldq, seq, headDim, kps, vps, args, workers)
}

// attendHeads runs every query head's attention over packed key/value heads,
// query head h reading group h/(heads/len(kps)).
func attendHeads(out []float32, ldOut int, q []float32, ldq, seq, headDim int, kps, vps []Packed, args AttentionArgs, workers int) {
	heads := args.Heads
	group := heads / len(kps)
	if seq == 1 && group > 1 {
		// One query per head (a decoding step): a group's queries are
		// consecutive in q, so they go through each product together, the
		// keys and values read once. The kernels sum each row the same way
		// whatever the number of rows, so this is bit-identical to one head
		// at a time.
		forEach(len(kps), workers, func(g int) {
			off := g * group * headDim
			attendRows(out[off:], headDim, q[off:], headDim, group, &kps[g], &vps[g], g*group, 0, args)
		})
		return
	}
	forEach(heads, workers, func(h int) {
		off := h * headDim
		attendRows(out[off:], ldOut, q[off:], ldq, seq, &kps[h/group], &vps[h/group], h*seq, 1, args)
	})
}

// attendRows computes attention for rows queries (row r at q[r*ldq]) over
// keys kp (packed for q·kᵀ) and values vp (packed for probs·v), writing row
// r's context to out[r*ldOut:]. Row r uses row biasRow0+r of args.Bias and,
// with args.Causal, sees keys up to args.Offset+queryStep·r: queryStep is 1
// when the rows are successive queries of one head, 0 when they are heads of
// one query.
func attendRows(out []float32, ldOut int, q []float32, ldq, rows int, kp, vp *Packed, biasRow0, queryStep int, args AttentionArgs) {
	kvSeq := kp.n
	buf := getScratch(rows * kvSeq)
	defer scratch.Put(buf)
	scores := *buf
	gemm(rows, q, ldq, kp, nil, scores, kvSeq, 1)
	for r := range rows {
		row := scores[r*kvSeq : (r+1)*kvSeq]
		scale := args.Scale
		if args.Bias != nil {
			// The bias is added after scaling, so scale here and let the
			// softmax run unscaled.
			bias := args.Bias[(biasRow0+r)*kvSeq : (biasRow0+r+1)*kvSeq]
			for j := range row {
				row[j] = row[j]*scale + bias[j]
			}
			scale = 1
		}
		if args.Causal {
			for j := args.Offset + queryStep*r + 1; j < kvSeq; j++ {
				row[j] = float32(math.Inf(-1))
			}
		}
		softmaxScaled(row, scale)
	}
	gemm(rows, scores, kvSeq, vp, nil, out, ldOut, 1)
}

// KVCache holds attention keys and values packed per head, for attention
// that reuses them: cross-attention over a fixed encoder output, or a
// decoder's self-attention cache, which grows one position at a time.
// Appending writes straight into the packed layout, so attending to the
// cache never repacks it. Results are bit-identical to AttentionStrided on
// the same keys and values.
type KVCache struct {
	heads, headDim int
	n, capacity    int
	keys           []Packed // per head: kp for q·kᵀ, n columns (positions) over headDim
	values         []Packed // per head: vp for probs·v, headDim columns over n positions
}

// NewKVCache returns an empty cache for heads of headDim, with room for
// capacity positions before it has to grow.
func NewKVCache(heads, headDim, capacity int) *KVCache {
	c := &KVCache{heads: heads, headDim: headDim}
	c.keys, c.values = make([]Packed, heads), make([]Packed, heads)
	c.reserve(max(capacity, 1))
	return c
}

// Len returns the number of positions in the cache.
func (c *KVCache) Len() int { return c.n }

// reserve grows the cache to hold capacity positions. Key panels hold 16
// positions each, so more of them are appended; value panels hold 16 of the
// headDim columns over all positions, so they are laid out again with the
// longer stride.
func (c *KVCache) reserve(capacity int) {
	if capacity <= c.capacity {
		return
	}
	for h := range c.heads {
		keys := make([]float32, packedSize(capacity, c.headDim))
		copy(keys, c.keys[h].data)
		c.keys[h] = Packed{n: c.n, k: c.headDim, data: keys}

		vPanels := (c.headDim + nr - 1) / nr
		values := make([]float32, vPanels*capacity*nr)
		for p := range vPanels {
			if c.capacity > 0 {
				copy(values[p*capacity*nr:], c.values[h].data[p*c.capacity*nr:p*c.capacity*nr+c.n*nr])
			}
		}
		c.values[h] = Packed{n: c.headDim, k: c.n, data: values, stride: capacity * nr}
	}
	c.capacity = capacity
}

// Append adds count positions: keys row i at k[i*ldk], values row i at
// v[i*ldv], each heads·headDim wide. The cache doubles when full.
func (c *KVCache) Append(k []float32, ldk int, v []float32, ldv int, count int) {
	if c.n+count > c.capacity {
		c.reserve(max(2*c.capacity, c.n+count))
	}
	hd := c.headDim
	for i := range count {
		pos := c.n + i
		kRow, vRow := k[i*ldk:i*ldk+c.heads*hd], v[i*ldv:i*ldv+c.heads*hd]
		for h := range c.heads {
			keys := c.keys[h].data[(pos/nr)*hd*nr+pos%nr:]
			for d, x := range kRow[h*hd : (h+1)*hd] {
				keys[d*nr] = x
			}
			values := c.values[h].data
			for d, x := range vRow[h*hd : (h+1)*hd] {
				values[(d/nr)*c.capacity*nr+pos*nr+d%nr] = x
			}
		}
	}
	c.n += count
	for h := range c.heads {
		c.keys[h].n = c.n
		c.values[h].k = c.n
	}
}

// Attend computes attention for seq queries (row i at q[i*ldq],
// args.Heads·headDim wide) over every cached position, writing the context
// for query i to out[i*ldOut:]. The cache holds the key/value heads:
// args.KVHeads (or, when 0, args.Heads) must match it. With args.Causal, the
// queries are positions args.Offset.. of the cache.
func (c *KVCache) Attend(out []float32, ldOut int, q []float32, ldq int, seq int, args AttentionArgs) {
	args.checkHeads("KVCache.Attend")
	if args.kvHeads() != c.heads {
		panic(fmt.Sprintf("ops.KVCache.Attend: %d key/value heads, cache has %d", args.kvHeads(), c.heads))
	}
	if args.Bias != nil && len(args.Bias) != args.Heads*seq*c.n {
		panic(fmt.Sprintf("ops.KVCache.Attend: bias has %d values, want %d·%d·%d", len(args.Bias), args.Heads, seq, c.n))
	}
	if args.Causal && args.Offset+seq > c.n {
		panic(fmt.Sprintf("ops.KVCache.Attend: %d causal queries after %d positions need %d keys, have %d", seq, args.Offset, args.Offset+seq, c.n))
	}
	attendHeads(out, ldOut, q, ldq, seq, c.headDim, c.keys, c.values, args, workersFor(2*seq*c.n*args.Heads*c.headDim))
}

// RMSNorm scales each row of x in place to unit root mean square, then by
// the per-feature weight: T5's layer norm, which subtracts no mean and adds
// no bias. Statistics are accumulated in float64.
func RMSNorm(x, weight *tensor.Tensor, eps float64) {
	cols := x.Cols()
	if weight.Size() != cols {
		panic(fmt.Sprintf("ops.RMSNorm: x is %v, weight %v", x.Shape, weight.Shape))
	}
	for i := range x.Rows() {
		row := x.Row(i)
		var sum float64
		for _, v := range row {
			sum += float64(v) * float64(v)
		}
		inv := 1 / math.Sqrt(sum/float64(cols)+eps)
		for j, v := range row {
			row[j] = weight.Data[j] * float32(float64(v)*inv)
		}
	}
}

// ReLU replaces negative values of x with zero, in place.
func ReLU(x []float32) {
	for i, v := range x {
		x[i] = max(v, 0)
	}
}

// SiLU applies x·sigmoid(x) in place, computed as x/(1+e^-x) like PyTorch.
func SiLU(x []float32) {
	for i, v := range x {
		x[i] = silu(v)
	}
}

func silu(v float32) float32 { return v / (1 + expf(-v)) }

// SwiGLUInto sets out [rows, ff] to SiLU(gate)·up, the gated feed-forward
// activation of Llama-family models, where gateUp [rows, 2·ff] holds the
// gate projection's output in its first ff columns and the up projection's
// in the rest (one Linear over both weights).
func SwiGLUInto(out, gateUp *tensor.Tensor) {
	ff := out.Cols()
	if gateUp.Rows() != out.Rows() || gateUp.Cols() != 2*ff {
		panic(fmt.Sprintf("ops.SwiGLUInto: out is %v, gateUp %v", out.Shape, gateUp.Shape))
	}
	for i := range out.Rows() {
		gu, o := gateUp.Row(i), out.Row(i)
		gate, up := gu[:ff], gu[ff:]
		for j, g := range gate {
			o[j] = silu(g) * up[j]
		}
	}
}

// Argmax returns the index of the largest value of x, the first one on ties.
// x must not be empty.
func Argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// Gather returns the rows of table selected by ids, as a new [len(ids), cols]
// tensor. Callers validate ids; an out-of-range id panics.
func Gather(table *tensor.Tensor, ids []int) *tensor.Tensor {
	out := tensor.New(len(ids), table.Cols())
	for i, id := range ids {
		copy(out.Row(i), table.Row(id))
	}
	return out
}
