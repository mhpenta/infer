package ops

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

// BenchmarkMicroLinear is one Inkling 2 Mini expert at decode: the gate and
// up projections [2304, 2304] and the down projection [2304, 1152], NVFP4,
// for one position.
func BenchmarkMicroLinear(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	t := &testing.T{}
	gu := randomMicro(t, rng, NVFP4, 2304, 2304)
	down := randomMicro(t, rng, NVFP4, 2304, 1152)
	x := make([]float32, 2304)
	y := make([]float32, 2304)
	for i := range x {
		x[i] = rng.Float32()
	}
	b.SetBytes(int64(len(gu.values) + len(down.values)))
	for b.Loop() {
		MicroLinearInto(y, 2304, x, 2304, 1, gu)
		MicroLinearInto(y, 2304, x, 1152, 1, down)
	}
}

// BenchmarkMicroRows compares the two ways to multiply n rows by an expert's
// gate and up projections: decoding blocks as it goes, or dequantizing the
// matrix and running the dense product.
func BenchmarkMicroRows(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	w := randomMicro(b, rng, NVFP4, 2304, 2304)
	for _, n := range []int{2, 4, 8, 16} {
		x := tensor.New(n, 2304)
		for i := range x.Data {
			x.Data[i] = rng.Float32()
		}
		y := make([]float32, n*2304)
		b.Run(fmt.Sprintf("micro/%d", n), func(b *testing.B) {
			for b.Loop() {
				MicroLinearInto(y, 2304, x.Data, 2304, n, w)
			}
		})
		b.Run(fmt.Sprintf("dense/%d", n), func(b *testing.B) {
			dense := tensor.New(2304, 2304)
			for b.Loop() {
				w.Dequantize(dense.Data, 0)
				MatMulT(x, dense)
			}
		})
	}
}
