// Package pool turns per-token hidden states into a single sentence vector.
package pool

import (
	"math"

	"github.com/mhpenta/infer/tensor"
)

// Mean averages the rows of a [seq, hidden] tensor. Every row must be a real
// token: this is masked mean pooling for an unpadded sequence, matching
// sentence-transformers' mean pooling on a batch of one.
func Mean(hidden *tensor.Tensor) []float32 {
	rows, cols := hidden.Rows(), hidden.Cols()
	sums := make([]float64, cols)
	for i := range rows {
		for j, v := range hidden.Row(i) {
			sums[j] += float64(v)
		}
	}
	out := make([]float32, cols)
	for j, s := range sums {
		out[j] = float32(s / float64(rows))
	}
	return out
}

// CLS returns a copy of the first row of a [seq, hidden] tensor: the hidden
// state of the [CLS] token, as sentence-transformers' CLS pooling uses.
func CLS(hidden *tensor.Tensor) []float32 {
	return append([]float32(nil), hidden.Row(0)...)
}

// L2Normalize scales v to unit length in place, accumulating in float64. A
// zero vector is left unchanged.
func L2Normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := 1 / math.Sqrt(sum)
	for i, x := range v {
		v[i] = float32(float64(x) * inv)
	}
}
