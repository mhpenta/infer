package ops

import (
	"math/rand/v2"
	"slices"
	"testing"
)

// TestKernelBoundRejectsWrongKernels checks that kernelOutputsAgree, which
// the AVX-512 startup self-check and TestKernelMatchesPortable use, still
// catches broken kernels at the self-check's depths, and still accepts a
// correct kernel that sums in another order. The broken kernels are the
// portable one given wrong inputs, standing for the usual microkernel bugs.
func TestKernelBoundRejectsWrongKernels(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	const ldc = 2*nr + 3
	for _, k := range []int{1, 2, 3, 4, 7, 256, 301} {
		a0, a1 := randomFloats(r, k*mr), randomFloats(r, k*mr)
		b0, b1 := randomFloats(r, k*nr), randomFloats(r, k*nr)
		q0, q1 := randomInt8(r, k*nr), randomInt8(r, k*nr)
		bias, start := randomFloats(r, 2*nr), randomFloats(r, 2*mr*ldc)
		wide := func(a0, a1, b0, b1 []float32, k, mode int) []float32 {
			c := slices.Clone(start)
			kernel12x32Portable(&a0[0], &a1[0], &b0[0], &b1[0], k, &c[0], ldc, &bias[0], mode)
			return c
		}
		// reversed returns a strip or panel with its k steps in reverse order.
		reversed := func(s []float32, width int) []float32 {
			out := make([]float32, 0, len(s))
			for p := k - 1; p >= 0; p-- {
				out = append(out, s[p*width:(p+1)*width]...)
			}
			return out
		}
		for _, mode := range []int{storeC, storeBiasC, addC} {
			want := wide(a0, a1, b0, b1, k, mode)
			strips, panels := [][]float32{a0, a1}, [][]float32{b0, b1}

			// Summing the same products in reverse order is correct.
			got := wide(reversed(a0, mr), reversed(a1, mr), reversed(b0, nr), reversed(b1, nr), k, mode)
			if err := kernelOutputsAgree(got, want, start, ldc, strips, panels, k, bias, mode); err != nil {
				t.Errorf("k=%d mode=%d: a kernel summing in reverse order was rejected: %v", k, mode, err)
			}

			shifted := slices.Clone(b0) // column j reads column j+1
			for p := range k {
				row := shifted[p*nr : (p+1)*nr]
				row = append(row[1:], row[0])
				copy(shifted[p*nr:], row)
			}
			flipped := slices.Clone(b1) // one product's sign
			flipped[(k/2)*nr+5] = -flipped[(k/2)*nr+5]
			wrong := map[string][]float32{
				"panels swapped":         wide(a0, a1, b1, b0, k, mode),
				"strips swapped":         wide(a1, a0, b0, b1, k, mode),
				"panel shifted a column": wide(a0, a1, shifted, b1, k, mode),
				"one product negated":    wide(a0, a1, b0, flipped, k, mode),
			}
			if k > 1 {
				wrong["last k step dropped"] = wide(a0, a1, b0, b1, k-1, mode)
				wrong["first k step dropped"] = wide(a0[mr:], a1[mr:], b0[nr:], b1[nr:], k-1, mode)
			}
			if mode != storeC {
				wrong["epilogue as storeC"] = wide(a0, a1, b0, b1, k, storeC)
			}
			for name, got := range wrong {
				if kernelOutputsAgree(got, want, start, ldc, strips, panels, k, bias, mode) == nil {
					t.Errorf("k=%d mode=%d: a kernel with %s was accepted", k, mode, name)
				}
			}

			// int8 panels widened as unsigned bytes.
			wantI, gotI := slices.Clone(start), slices.Clone(start)
			kernel12x32PortableInt8(&a0[0], &a1[0], &q0[0], &q1[0], k, &wantI[0], ldc, &bias[0], mode)
			u0, u1 := widenInt8Panel(q0), widenInt8Panel(q1)
			for i := range u0 {
				u0[i], u1[i] = float32(uint8(q0[i])), float32(uint8(q1[i]))
			}
			kernel12x32Portable(&a0[0], &a1[0], &u0[0], &u1[0], k, &gotI[0], ldc, &bias[0], mode)
			if kernelOutputsAgree(gotI, wantI, start, ldc, strips, [][]float32{widenInt8Panel(q0), widenInt8Panel(q1)}, k, bias, mode) == nil {
				t.Errorf("k=%d mode=%d: an int8 kernel widening unsigned was accepted", k, mode)
			}
		}
	}
}

// TestKernelBoundChecksOutsideTile checks that a kernel writing outside its
// tile is caught even when the value written is close.
func TestKernelBoundChecksOutsideTile(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 54))
	const k, ldc = 4, nr + 3
	a, b := randomFloats(r, k*mr), randomFloats(r, k*nr)
	bias, start := randomFloats(r, nr), randomFloats(r, mr*ldc)
	want := slices.Clone(start)
	kernel6x16Portable(&a[0], &b[0], k, &want[0], ldc, &bias[0], storeC)
	got := slices.Clone(want)
	got[nr] += 1e-7 // the first padding element of row 0
	if kernelOutputsAgree(got, want, start, ldc, [][]float32{a}, [][]float32{b}, k, bias, storeC) == nil {
		t.Error("a write past the tile's last column was accepted")
	}
}
