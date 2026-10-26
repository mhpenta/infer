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
	attention(out.Data, q.Data, k.Data, v.Data, hidden, hidden, seq, kvSeq, hidden, heads, scale)
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
	attention(out.Data, d, d[hidden:], d[2*hidden:], 3*hidden, 3*hidden, seq, seq, hidden, heads, scale)
}

// attention computes multi-head attention on strided views: row i of q at
// q[i*ldq], rows of k and v at multiples of ldkv, rows of out at multiples of
// hidden. Heads run in parallel; each head's two products run serially on
// views of its columns, and only k and v are copied, into panels.
func attention(out, q, k, v []float32, ldq, ldkv, seq, kvSeq, hidden, heads int, scale float32) {
	headDim := hidden / heads
	forEach(heads, workersFor(2*seq*kvSeq*hidden), func(h int) {
		off := h * headDim
		kSize, vSize := packedSize(kvSeq, headDim), packedSize(headDim, kvSeq)
		buf := getScratch(seq*kvSeq + kSize + vSize)
		defer scratch.Put(buf)
		scores, kBuf, vBuf := (*buf)[:seq*kvSeq], (*buf)[seq*kvSeq:seq*kvSeq+kSize], (*buf)[seq*kvSeq+kSize:]
		kp := packTInto(kBuf, k[off:], kvSeq, headDim, ldkv)
		gemm(seq, q[off:], ldq, &kp, nil, scores, kvSeq, 1)
		for i := range seq {
			softmaxScaled(scores[i*kvSeq:(i+1)*kvSeq], scale)
		}
		vp := packNInto(vBuf, v[off:], kvSeq, headDim, ldkv)
		gemm(seq, scores, kvSeq, &vp, nil, out[off:], hidden, 1)
	})
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
