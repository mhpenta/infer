package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/tensor"
)

// halfWeights returns an n×k weight whose values are exact in fp16.
func halfWeights(r *rand.Rand, n, k int) *tensor.Tensor {
	w := tensor.New(n, k)
	for i, h := range randomHalves(r, n*k) {
		w.Data[i] = half.ToFloat32(h)
	}
	return w
}

// TestPackHalfMatchesPack checks that fp16 panels give the same products as
// float32 panels. With SIMD kernels the widened values and the FMA order are
// identical, so the results must be bit for bit equal; the portable fp16
// kernel is only required to be close.
func TestPackHalfMatchesPack(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, forceWide := range []bool{false, true} {
		saved := wide
		wide = wide || forceWide
		for _, m := range []int{1, 3, 6, 13, 24, 65, 150} {
			for _, n := range []int{16, 17, 32, 100} {
				for _, k := range []int{1, 7, 256, 300} {
					name := fmt.Sprintf("wide=%v/%dx%dx%d", wide, m, n, k)
					x := tensor.FromData(randomSlice(r, m*k), m, k)
					w := halfWeights(r, n, k)
					b := tensor.FromData(randomSlice(r, n), n)
					hp, err := PackHalf(w)
					if err != nil {
						t.Fatal(err)
					}
					if !hp.Half() {
						t.Fatal("PackHalf did not store fp16 panels")
					}
					got, want := Linear(x, hp, b), Linear(x, Pack(w), b)
					if HalfPanels() {
						if !slices.Equal(got.Data, want.Data) {
							t.Fatalf("%s: fp16 panels differ from float32 panels", name)
						}
						continue
					}
					for i := range got.Data {
						if math.Abs(float64(got.Data[i]-want.Data[i])) > 1e-4*(1+math.Abs(float64(want.Data[i]))) {
							t.Fatalf("%s: element %d = %g, float32 panels give %g", name, i, got.Data[i], want.Data[i])
						}
					}
				}
			}
		}
		wide = saved
	}
}

func TestPackHalfRejectsInexact(t *testing.T) {
	w := tensor.FromData([]float32{1, 0.1}, 1, 2)
	if _, err := PackHalf(w); err == nil {
		t.Error("PackHalf accepted 0.1, which fp16 cannot hold exactly")
	}
}

// TestKernelHalfMatchesPortable checks the active fp16-panel kernel against
// the portable one in every epilogue mode.
func TestKernelHalfMatchesPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 44))
	const ldc = nr + 5
	for _, k := range []int{1, 3, 4, 5, 8, 255, 300} {
		a, b := randomSlice(r, k*mr), randomHalves(r, k*nr)
		bias, start := randomSlice(r, nr), randomSlice(r, mr*ldc)
		for _, mode := range []int{storeC, storeBiasC, addC} {
			got, want := slices.Clone(start), slices.Clone(start)
			kernel6x16Half(&a[0], &b[0], k, &got[0], ldc, &bias[0], mode)
			kernel6x16PortableHalf(&a[0], &b[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnoughTest(got, want); err != nil {
				t.Fatalf("%s kernel, k=%d, mode %d: %v", Kernel(), k, mode, err)
			}
		}
	}
}

func closeEnoughTest(got, want []float32) error {
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
			return fmt.Errorf("c[%d] = %g, portable %g", i, got[i], want[i])
		}
	}
	return nil
}
