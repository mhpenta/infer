package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

// int8Weights returns an n×k weight of normal draws with an outlier in some
// rows, as real weights have, so that rows get very different scales.
func int8Weights(r *rand.Rand, n, k int) *tensor.Tensor {
	w := tensor.FromData(randomSlice(r, n*k), n, k)
	for j := 0; j < n; j += 3 {
		w.Row(j)[r.IntN(k)] *= 20
	}
	if n > 1 {
		clear(w.Row(1)) // a row of zeros
	}
	return w
}

// int8Reference computes x·wᵀ (+ b) as PackInt8 defines it: the product with
// float32 panels holding each row's integers, then each column scaled and
// the bias added.
func int8Reference(x, w, b *tensor.Tensor) *tensor.Tensor {
	n, k := w.Rows(), w.Cols()
	ints := tensor.New(n, k)
	scale := make([]float32, n)
	q := make([]int8, k)
	for j := range n {
		scale[j] = quantizeRow(q, w.Row(j))
		for kk, v := range q {
			ints.Row(j)[kk] = float32(v)
		}
	}
	y := Linear(x, Pack(ints), nil)
	for i := range y.Rows() {
		for j, v := range y.Row(i) {
			if b == nil {
				y.Row(i)[j] = v * scale[j]
			} else {
				y.Row(i)[j] = float32(v*scale[j]) + b.Data[j]
			}
		}
	}
	return y
}

// TestPackInt8MatchesIntegerProduct checks int8 panels against float32
// panels holding the same integers, followed by the per-column scaling. With
// SIMD kernels the widened values and the FMA order are identical, so results
// must be bit for bit equal; the portable kernels are only required to be
// close.
func TestPackInt8MatchesIntegerProduct(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 72))
	for _, forceWide := range []bool{false, true} {
		saved := wide
		wide = wide || forceWide
		for _, m := range []int{1, 3, 6, 13, 24, 65, 150} {
			for _, n := range []int{16, 17, 32, 100} {
				for _, k := range []int{1, 7, 256, 300} {
					name := fmt.Sprintf("wide=%v/%dx%dx%d", wide, m, n, k)
					x := tensor.FromData(randomSlice(r, m*k), m, k)
					w := int8Weights(r, n, k)
					p := PackInt8(w)
					if !p.Int8() || p.BF16() || p.Half() {
						t.Fatalf("%s: packed format %v", name, p.format)
					}
					for _, b := range []*tensor.Tensor{nil, tensor.FromData(randomSlice(r, n), n)} {
						got, want := Linear(x, p, b), int8Reference(x, w, b)
						if Int8Panels() {
							if !slices.Equal(got.Data, want.Data) {
								t.Fatalf("%s bias=%v: int8 panels differ from the integer product", name, b != nil)
							}
							continue
						}
						if err := closeEnoughTest(got.Data, want.Data); err != nil {
							t.Fatalf("%s bias=%v: %v", name, b != nil, err)
						}
					}
				}
			}
		}
		wide = saved
	}
}

// TestQuantizeRow checks the rounding: every value within half a scale step,
// the largest magnitude at ±127, and zeros kept exact.
func TestQuantizeRow(t *testing.T) {
	r := rand.New(rand.NewPCG(73, 74))
	for _, row := range [][]float32{
		randomSlice(r, 300),
		{0, 0, 0},
		{-2, 1, 0.5},
		{1e-30, -3e-30, 0},
		{5},
	} {
		q := make([]int8, len(row))
		scale := quantizeRow(q, row)
		var peak float32
		for _, v := range row {
			peak = max(peak, float32(math.Abs(float64(v))))
		}
		if peak == 0 {
			if scale != 0 || slices.ContainsFunc(q, func(v int8) bool { return v != 0 }) {
				t.Errorf("zero row: scale %g, q %v", scale, q)
			}
			continue
		}
		if !slices.Contains(q, 127) && !slices.Contains(q, -127) {
			t.Errorf("row %v: no value at ±127 in %v", row[:min(3, len(row))], q[:min(3, len(q))])
		}
		for kk, v := range row {
			if q[kk] == -128 {
				t.Fatalf("q[%d] = -128", kk)
			}
			if d := math.Abs(float64(v) - float64(scale)*float64(q[kk])); d > 0.5*float64(scale)*(1+1e-6) {
				t.Errorf("row[%d] = %g, quantized %d × %g, off by %.3g steps", kk, v, q[kk], scale, d/float64(scale))
			}
		}
	}
}

// TestKernelInt8MatchesPortable checks the active int8-panel kernels against
// the portable ones in every epilogue mode.
func TestKernelInt8MatchesPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(75, 76))
	const ldc = 2*nr + 5
	for _, k := range []int{1, 2, 3, 4, 5, 8, 255, 300} {
		a0, a1 := randomSlice(r, k*mr), randomSlice(r, k*mr)
		b0, b1 := randomInt8(r, k*nr), randomInt8(r, k*nr)
		bias, start := randomSlice(r, 2*nr), randomSlice(r, 2*mr*ldc)
		for _, mode := range []int{storeC, storeBiasC, addC} {
			got, want := slices.Clone(start), slices.Clone(start)
			kernel6x16Int8(&a0[0], &b0[0], k, &got[0], ldc, &bias[0], mode)
			kernel6x16PortableInt8(&a0[0], &b0[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnoughTest(got, want); err != nil {
				t.Fatalf("%s 6x16 kernel, k=%d, mode %d: %v", Kernel(), k, mode, err)
			}
			got, want = slices.Clone(start), slices.Clone(start)
			kernel12x32Int8(&a0[0], &a1[0], &b0[0], &b1[0], k, &got[0], ldc, &bias[0], mode)
			kernel12x32PortableInt8(&a0[0], &a1[0], &b0[0], &b1[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnoughTest(got, want); err != nil {
				t.Fatalf("%s 12x32 kernel, k=%d, mode %d: %v", Kernel(), k, mode, err)
			}
		}
		for _, panels := range []int{1, 4} {
			b := randomInt8(r, 4*k*nr)
			var got, want [4 * nr]float32
			gemvPanelsInt8(&a0[0], &b[0], k*nr, k, panels, &got)
			gemvGoInt8(&a0[0], &b[0], k*nr, k, panels, want[:])
			if err := closeEnoughTest(got[:panels*nr], want[:panels*nr]); err != nil {
				t.Fatalf("%s gemv, %d panels, k=%d: %v", Kernel(), panels, k, err)
			}
		}
	}
}

// TestQuantizeInt8 checks that the row-major quantization holds the same
// integers and scales as PackInt8's panels.
func TestQuantizeInt8(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, sh := range [][2]int{{1, 5}, {37, 100}, {130, 64}} {
		n, k := sh[0], sh[1]
		w := int8Weights(r, n, k)
		q, scale := QuantizeInt8(w)
		p := PackInt8(w)
		if !slices.Equal(scale, p.scale) {
			t.Fatalf("%dx%d: scales differ from PackInt8's", n, k)
		}
		for j := range n {
			for kk := range k {
				if got, want := q[j*k+kk], p.quant[(j/nr)*k*nr+kk*nr+j%nr]; got != want {
					t.Fatalf("%dx%d: q[%d,%d] = %d, PackInt8 has %d", n, k, j, kk, got, want)
				}
			}
		}
	}
}
