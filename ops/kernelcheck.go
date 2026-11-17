package ops

import (
	"fmt"
	"math"

	"github.com/mhpenta/infer/internal/half"
)

// kernelTolerance is c in the bound kernelOutputsAgree allows between two
// microkernels: c·(k+1)·ε·(Σₚ|aₚ·bₚ| + |e|) per output, where ε is float32's
// machine epsilon and e is what the epilogue adds (the bias, or c's prior
// value with addC).
//
// Two kernels may sum in different orders, or one may fuse each multiply-add
// where the other rounds the product first. Either way, a sum of k products
// in float32 is within k·ε·Σ|aₚ·bₚ| of the exact sum (each step rounds at
// most twice, by ε/2 each, scaling a partial sum no larger than Σ|aₚ·bₚ|),
// and the epilogue adds one more rounding of at most ε/2 of the result. Two
// kernels can each be that far off in opposite directions, so c = 2 bounds
// any correct pair on any data. A fixed relative tolerance does not: when
// the products are large and cancel, the error is large next to the result.
// int8 panels (values up to ±127) at k=301 put one output 14,000 ulps from
// the portable kernel's, a relative error of 1.4e-3, well inside this bound.
const kernelTolerance = 2

// kernelOutputsAgree compares got, a kernel's output, with want, the
// portable kernel's, for one microkernel call: a tile of len(strips)·mr rows
// by len(panels)·nr columns at the start of c, rows ldc apart, computed from
// packed strips (k steps of mr values each) and panels widened to float32
// (k steps of nr values each). start is c before the call; outputs outside
// the tile must still hold it, bit for bit.
func kernelOutputsAgree(got, want, start []float32, ldc int, strips, panels [][]float32, k int, bias []float32, mode int) error {
	rows, cols := len(strips)*mr, len(panels)*nr
	inTile := func(i int) bool { return i/ldc < rows && i%ldc < cols }
	for i := range got {
		if !inTile(i) {
			if math.Float32bits(got[i]) != math.Float32bits(start[i]) {
				return fmt.Errorf("c[%d], outside the %d×%d tile, changed from %g to %g", i, rows, cols, start[i], got[i])
			}
			continue
		}
		r, j := i/ldc, i%ldc
		a, b := strips[r/mr], panels[j/nr]
		var sum float64
		for p := range k {
			sum += math.Abs(float64(a[p*mr+r%mr]) * float64(b[p*nr+j%nr]))
		}
		switch mode {
		case storeBiasC:
			sum += math.Abs(float64(bias[j]))
		case addC:
			sum += math.Abs(float64(start[i]))
		}
		bound := kernelTolerance * float64(k+1) * 0x1p-23 * sum
		if d := math.Abs(float64(got[i]) - float64(want[i])); !(d <= bound) {
			return fmt.Errorf("c[%d] (row %d, column %d) = %g, portable %g: differs by %.3g, more than the bound %.3g", i, r, j, got[i], want[i], d, bound)
		}
	}
	return nil
}

// widenPanel returns a panel of 16-bit values in format f as float32.
func widenPanel(f panelFormat, src []uint16) []float32 {
	out := make([]float32, len(src))
	for i, h := range src {
		if f == formatBF16 {
			out[i] = math.Float32frombits(uint32(h) << 16)
		} else {
			out[i] = half.ToFloat32(h)
		}
	}
	return out
}

// widenInt8Panel returns an int8 panel as float32.
func widenInt8Panel(src []int8) []float32 {
	out := make([]float32, len(src))
	for i, q := range src {
		out[i] = float32(q)
	}
	return out
}
