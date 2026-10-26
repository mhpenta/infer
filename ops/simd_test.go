package ops

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// simdLengths covers every tail size around the 8-lane width plus long rows.
var simdLengths = func() []int {
	var n []int
	for i := 1; i <= 70; i++ {
		n = append(n, i)
	}
	return append(n, 512, 1003)
}()

func closeRel(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol*(1+math.Abs(b))
}

// TestSliceHelpersMatchPortable checks the dispatched helpers (AVX2 where
// available) against the portable loops.
func TestSliceHelpersMatchPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(21, 22))
	for _, n := range simdLengths {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * 8)
		}

		if got, want := maxSlice(x), maxGo(x); got != want {
			t.Fatalf("n=%d: maxSlice = %g, maxGo = %g", n, got, want)
		}

		got, want := slices.Clone(x), slices.Clone(x)
		peak := maxGo(x)
		gotSum := expShiftScale(got, peak, 0.37)
		wantSum := expShiftScaleGo(want, peak, 0.37)
		if !closeRel(gotSum, wantSum, 1e-6) {
			t.Fatalf("n=%d: expShiftScale sum %g, portable %g", n, gotSum, wantSum)
		}
		for i := range got {
			if !closeRel(float64(got[i]), float64(want[i]), 1e-6) {
				t.Fatalf("n=%d: exp[%d] = %g, portable %g", n, i, got[i], want[i])
			}
		}

		got, want = slices.Clone(x), slices.Clone(x)
		scaleSlice(got, 0.25)
		scaleGo(want, 0.25)
		if !slices.Equal(got, want) {
			t.Fatalf("n=%d: scaleSlice differs from portable", n)
		}

		got, want = slices.Clone(x), slices.Clone(x)
		geluSlice(got)
		geluGo(want)
		for i := range got {
			if !closeRel(float64(got[i]), float64(want[i]), 1e-6) {
				t.Fatalf("n=%d: gelu[%d] of %g = %g, portable %g", n, i, x[i], got[i], want[i])
			}
		}
	}
}

func TestExpShiftScaleEdges(t *testing.T) {
	// Masked (-Inf) and very negative inputs give exactly 0, in vector lanes
	// and in the tail alike.
	x := make([]float32, 19)
	for i := range x {
		x[i] = float32(math.Inf(-1))
	}
	x[3], x[17] = 0, -200
	sum := expShiftScale(x, 0, 1)
	if sum != 1 {
		t.Errorf("sum = %g, want 1", sum)
	}
	for i, v := range x {
		want := float32(0)
		if i == 3 {
			want = 1
		}
		if v != want {
			t.Errorf("x[%d] = %g, want %g", i, v, want)
		}
	}
}

func TestSoftmaxScaledMatchesFloat64(t *testing.T) {
	r := rand.New(rand.NewPCG(23, 24))
	for _, n := range simdLengths {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64() * 10)
		}
		const scale = 0.17677669 // 1/√32, attention's scale for 32-wide heads
		want := make([]float64, n)
		peak := math.Inf(-1)
		for _, v := range x {
			peak = max(peak, float64(v)*scale)
		}
		var sum float64
		for i, v := range x {
			want[i] = math.Exp(float64(v)*scale - peak)
			sum += want[i]
		}
		got := slices.Clone(x)
		softmaxScaled(got, scale)
		for i := range got {
			if math.Abs(float64(got[i])-want[i]/sum) > 1e-6 {
				t.Fatalf("n=%d: softmax[%d] = %g, want %g", n, i, got[i], want[i]/sum)
			}
		}
	}
}

func BenchmarkSoftmaxRow(b *testing.B) {
	r := rand.New(rand.NewPCG(25, 26))
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(r.NormFloat64() * 10)
	}
	row := make([]float32, len(x))
	for b.Loop() {
		copy(row, x)
		softmaxScaled(row, 0.17677669)
	}
}

func BenchmarkGELUSlice(b *testing.B) {
	x := make([]float32, 1536*8)
	for i := range x {
		x[i] = float32(i%4096-2048) / 256
	}
	for b.Loop() {
		geluSlice(x)
	}
}
