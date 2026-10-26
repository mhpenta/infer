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

// TestGemvMatchesGemmExactly checks that a one-row product (gemv) gives
// bit-for-bit the values the tiled path gives for the same row. Two copies
// of the row force the tiled path.
func TestGemvMatchesGemmExactly(t *testing.T) {
	r := rand.New(rand.NewPCG(61, 62))
	for _, forceWide := range []bool{false, true} {
		saved := wide
		wide = wide || forceWide
		for _, n := range []int{1, 15, 16, 17, 64, 65, 100, 513} {
			for _, k := range []int{1, 2, 3, 255, 256, 257, 600} {
				for _, usePanels := range []string{"f32", "f16"} {
					name := fmt.Sprintf("wide=%v/n%d_k%d_%s", wide, n, k, usePanels)
					w := tensor.New(n, k)
					for i, h := range randomHalves(r, n*k) {
						w.Data[i] = half.ToFloat32(h)
					}
					p := Pack(w)
					if usePanels == "f16" {
						var err error
						if p, err = PackHalf(w); err != nil {
							t.Fatal(err)
						}
					}
					row := randomSlice(r, k)
					bias := randomSlice(r, n)
					two := append(slices.Clone(row), row...)
					for _, b := range [][]float32{nil, bias} {
						got := make([]float32, n)
						gemm(1, row, k, p, b, got, n, 1)
						want := make([]float32, 2*n)
						gemm(2, two, k, p, b, want, n, 1)
						for j := range n {
							if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
								t.Fatalf("%s bias=%v: c[%d] = %g, tiled path gives %g", name, b != nil, j, got[j], want[j])
							}
						}
					}
				}
			}
		}
		wide = saved
	}
}

func TestGemvParallelMatchesSerial(t *testing.T) {
	r := rand.New(rand.NewPCG(63, 64))
	n, k := 1000, 300
	p := Pack(tensor.FromData(randomSlice(r, n*k), n, k))
	row := randomSlice(r, k)
	serial, parallel := make([]float32, n), make([]float32, n)
	gemv(row, p, nil, serial, 1)
	gemv(row, p, nil, parallel, 5)
	if !slices.Equal(serial, parallel) {
		t.Fatal("parallel gemv differs from serial")
	}
}
