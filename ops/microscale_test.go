package ops

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestE4M3(t *testing.T) {
	for _, c := range []struct {
		b    byte
		want float32
	}{
		{0x00, 0}, {0x38, 1}, {0xB8, -1}, {0x7E, 448}, {0xFE, -448},
		{0x01, 1.0 / 512}, {0x08, 1.0 / 64}, {0x3C, 1.5}, {0x40, 2},
	} {
		if got := E4M3(c.b); got != c.want {
			t.Errorf("E4M3(%#x) = %g, want %g", c.b, got, c.want)
		}
	}
	if !math.IsNaN(float64(E4M3(0x7F))) || !math.IsNaN(float64(E4M3(0xFF))) {
		t.Error("E4M3 0x7F and 0xFF should be NaN")
	}
}

// randomMicro returns a random matrix in format f with finite values.
func randomMicro(t testing.TB, rng *rand.Rand, f MicroFormat, rows, cols int) *Microscaled {
	t.Helper()
	bs := f.BlockSize()
	finiteE4M3 := func() byte {
		for {
			if b := byte(rng.IntN(256)); b&0x7F != 0x7F {
				return b
			}
		}
	}
	var w *Microscaled
	var err error
	if f == NVFP4 {
		codes := make([]byte, rows*cols/2)
		for i := range codes {
			codes[i] = byte(rng.IntN(256))
		}
		scales := make([]byte, rows*cols/bs)
		for i := range scales {
			scales[i] = byte(0x20 + rng.IntN(0x30)) // positive, moderate
		}
		w, err = NewNVFP4(rows, cols, codes, scales, 0.0123)
	} else {
		values := make([]byte, rows*cols)
		for i := range values {
			values[i] = finiteE4M3()
		}
		scales := make([]byte, rows*cols/bs)
		for i := range scales {
			scales[i] = byte(115 + rng.IntN(10))
		}
		w, err = NewMXFP8(rows, cols, values, scales)
	}
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// refWeight is weight (r, k) computed from the format's definition.
func refWeight(w *Microscaled, r, k int) float64 {
	bs := w.format.BlockSize()
	block := (r*w.cols + k) / bs
	if w.format == NVFP4 {
		b := w.values[(r*w.cols+k)/2]
		code := b & 0xF
		if k%2 == 1 {
			code = b >> 4
		}
		return float64(e2m1[code]) * float64(E4M3(w.scales[block])) * float64(w.factor)
	}
	return float64(E4M3(w.values[r*w.cols+k])) * math.Ldexp(1, int(w.scales[block])-127)
}

func TestMicroscaledDequantizeAndProduct(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, f := range []MicroFormat{NVFP4, MXFP8} {
		rows, cols := 70, 96
		w := randomMicro(t, rng, f, rows, cols)

		dense := make([]float32, rows*cols)
		w.Dequantize(dense, 0)
		for r := range rows {
			for k := range cols {
				want := float32(refWeight(w, r, k))
				if got := dense[r*cols+k]; got != want {
					t.Fatalf("%v: weight (%d, %d) = %g, want %g", f, r, k, got, want)
				}
			}
		}

		for _, n := range []int{1, 3, 17, 40} {
			ldx, ldy := cols+5, rows+3
			x := make([]float32, n*ldx)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}
			y := make([]float32, n*ldy)
			MicroLinearInto(y, ldy, x, ldx, n, w)
			for i := range n {
				for r := range rows {
					var want, mag float64
					for k := range cols {
						p := float64(x[i*ldx+k]) * refWeight(w, r, k)
						want += p
						mag += math.Abs(p)
					}
					if d := math.Abs(float64(y[i*ldy+r]) - want); d > 1e-6*mag+1e-30 {
						t.Fatalf("%v n=%d: y[%d, %d] = %g, want %g", f, n, i, r, y[i*ldy+r], want)
					}
				}
			}
		}
	}
}

func TestMicroscaledRejectsBadShapes(t *testing.T) {
	if _, err := NewNVFP4(2, 24, make([]byte, 24), make([]byte, 3), 1); err == nil {
		t.Error("NVFP4 with a partial block was accepted")
	}
	if _, err := NewMXFP8(2, 32, make([]byte, 63), make([]byte, 2)); err == nil {
		t.Error("MXFP8 with a short value slice was accepted")
	}
}
