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

// narrowFormats are the 16-bit panel formats, with whether this build has
// SIMD kernels for each.
var narrowFormats = []struct {
	format panelFormat
	simd   func() bool
}{
	{formatF16, HalfPanels},
	{formatBF16, BF16Panels},
}

// randomNarrow returns n random values in format f (bits).
func randomNarrow(r *rand.Rand, f panelFormat, n int) []uint16 {
	if f == formatBF16 {
		return randomBF16(r, n)
	}
	return randomHalves(r, n)
}

// narrowWeights returns an n×k weight whose values are exact in format f.
func narrowWeights(r *rand.Rand, f panelFormat, n, k int) *tensor.Tensor {
	w := tensor.New(n, k)
	for i, h := range randomNarrow(r, f, n*k) {
		w.Data[i] = widenValue(f, h)
	}
	return w
}

// widenValue converts one 16-bit value in format f to float32.
func widenValue(f panelFormat, h uint16) float32 {
	if f == formatBF16 {
		return math.Float32frombits(uint32(h) << 16)
	}
	return half.ToFloat32(h)
}

// packNarrowAs packs w with the packer for format f.
func packNarrowAs(f panelFormat, w *tensor.Tensor) (*Packed, error) {
	if f == formatBF16 {
		return PackBF16(w)
	}
	return PackHalf(w)
}

// TestPackNarrowMatchesPack checks that fp16 and bf16 panels give the same
// products as float32 panels holding the widened values. With SIMD kernels
// the widened values and the FMA order are identical, so the results must be
// bit for bit equal; the portable kernels are only required to be close.
func TestPackNarrowMatchesPack(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, nf := range narrowFormats {
		for _, forceWide := range []bool{false, true} {
			saved := wide
			wide = wide || forceWide
			for _, m := range []int{1, 3, 6, 13, 24, 65, 150} {
				for _, n := range []int{16, 17, 32, 100} {
					for _, k := range []int{1, 7, 256, 300} {
						name := fmt.Sprintf("%v/wide=%v/%dx%dx%d", nf.format, wide, m, n, k)
						x := tensor.FromData(randomSlice(r, m*k), m, k)
						w := narrowWeights(r, nf.format, n, k)
						b := tensor.FromData(randomSlice(r, n), n)
						np, err := packNarrowAs(nf.format, w)
						if err != nil {
							t.Fatal(err)
						}
						if np.format != nf.format || np.Half() != (nf.format == formatF16) || np.BF16() != (nf.format == formatBF16) {
							t.Fatalf("%s: packed format %v", name, np.format)
						}
						got, want := Linear(x, np, b), Linear(x, Pack(w), b)
						if nf.simd() {
							if !slices.Equal(got.Data, want.Data) {
								t.Fatalf("%s: %v panels differ from float32 panels", name, nf.format)
							}
							continue
						}
						if err := closeEnoughTest(got.Data, want.Data); err != nil {
							t.Fatalf("%s: %v", name, err)
						}
					}
				}
			}
			wide = saved
		}
	}
}

func TestPackNarrowRejectsInexact(t *testing.T) {
	inexact := tensor.FromData([]float32{1, 0.1}, 1, 2)
	if _, err := PackHalf(inexact); err == nil {
		t.Error("PackHalf accepted 0.1, which fp16 cannot hold exactly")
	}
	if _, err := PackBF16(inexact); err == nil {
		t.Error("PackBF16 accepted 0.1, which bf16 cannot hold exactly")
	}
	// bf16 has float32's range: a tiny weight that fp16 would flush.
	tiny := tensor.FromData([]float32{math.Float32frombits(0x1e3c0000), 3}, 1, 2)
	if _, err := PackHalf(tiny); err == nil {
		t.Errorf("PackHalf accepted %g, which fp16 cannot hold", tiny.Data[0])
	}
	if _, err := PackBF16(tiny); err != nil {
		t.Errorf("PackBF16 rejected a bf16 value: %v", err)
	}
	if _, ok := half.FromFloat32Exact(tiny.Data[0]); ok {
		t.Fatal("test value is exact in fp16")
	}
}

// TestKernelNarrowMatchesPortable checks the active 16-bit-panel kernels
// against the portable ones in every epilogue mode.
func TestKernelNarrowMatchesPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 44))
	const ldc = nr + 5
	for _, nf := range narrowFormats {
		for _, k := range []int{1, 3, 4, 5, 8, 255, 300} {
			a, b := randomSlice(r, k*mr), randomNarrow(r, nf.format, k*nr)
			bias, start := randomSlice(r, nr), randomSlice(r, mr*ldc)
			for _, mode := range []int{storeC, storeBiasC, addC} {
				got, want := slices.Clone(start), slices.Clone(start)
				kernel6x16Narrow(nf.format, &a[0], &b[0], k, &got[0], ldc, &bias[0], mode)
				kernel6x16PortableNarrow(nf.format, &a[0], &b[0], k, &want[0], ldc, &bias[0], mode)
				if err := closeEnoughTest(got, want); err != nil {
					t.Fatalf("%s kernel, %v panels, k=%d, mode %d: %v", Kernel(), nf.format, k, mode, err)
				}
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
