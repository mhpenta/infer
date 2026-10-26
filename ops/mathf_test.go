package ops

import (
	"math"
	"testing"
)

// ulps returns how many float32 steps got is from the exact value want.
func ulps(got float32, want float64) float64 {
	w := float32(want)
	if got == w {
		return 0
	}
	step := math.Abs(float64(math.Nextafter32(w, float32(math.Inf(1))) - w))
	return math.Abs(float64(got)-want) / step
}

func TestExpf(t *testing.T) {
	var worst, worstX float64
	for x := -87.0; x <= 88.5; x += 0.000731 {
		got := expf(float32(x))
		if u := ulps(got, math.Exp(float64(float32(x)))); u > worst {
			worst, worstX = u, x
		}
	}
	if worst > 2 {
		t.Errorf("expf is %.2f ulps off at x=%g, want <= 2", worst, worstX)
	}
	t.Logf("max error %.2f ulps", worst)

	tests := []struct {
		in   float32
		want float32
	}{
		{0, 1},
		{float32(math.Inf(-1)), 0},
		{-100, 0},
		{100, float32(math.Inf(1))},
	}
	for _, tt := range tests {
		if got := expf(tt.in); got != tt.want {
			t.Errorf("expf(%g) = %g, want %g", tt.in, got, tt.want)
		}
	}
	if v := expf(float32(math.NaN())); v == v {
		t.Errorf("expf(NaN) = %g", v)
	}
}

func TestErff(t *testing.T) {
	// erf is near 0 for small x, so measure absolute error against 1 ulp of
	// the largest values (erf → 1) as well as relative error away from 0.
	var worstAbs, worstAbsX float64
	for x := -6.0; x <= 6.0; x += 0.000377 {
		got := float64(erff(float32(x)))
		want := math.Erf(float64(float32(x)))
		if d := math.Abs(got - want); d > worstAbs {
			worstAbs, worstAbsX = d, x
		}
	}
	if worstAbs > 4e-7 {
		t.Errorf("erff is %g off at x=%g, want <= 4e-7", worstAbs, worstAbsX)
	}
	t.Logf("max absolute error %.2g", worstAbs)
}

func TestGELUMatchesFloat64(t *testing.T) {
	var worst float64
	for x := -10.0; x <= 10.0; x += 0.00123 {
		v := []float32{float32(x)}
		GELU(v)
		f := float64(float32(x))
		want := 0.5 * f * (1 + math.Erf(f/math.Sqrt2))
		worst = max(worst, math.Abs(float64(v[0])-want)/(1+math.Abs(want)))
	}
	if worst > 1e-6 {
		t.Errorf("GELU relative error %g, want <= 1e-6", worst)
	}
}

func BenchmarkExp(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = -float32(i) / 256
	}
	b.Run("math.Exp", func(b *testing.B) {
		for b.Loop() {
			for _, v := range x {
				_ = float32(math.Exp(float64(v)))
			}
		}
	})
	b.Run("expf", func(b *testing.B) {
		for b.Loop() {
			for _, v := range x {
				_ = expf(v)
			}
		}
	})
}

func BenchmarkGELU(b *testing.B) {
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32(i-2048) / 256
	}
	b.Run("math.Erf", func(b *testing.B) {
		for b.Loop() {
			for _, v := range x {
				f := float64(v)
				_ = float32(0.5 * f * (1 + math.Erf(f/math.Sqrt2)))
			}
		}
	})
	b.Run("erff", func(b *testing.B) {
		for b.Loop() {
			for _, v := range x {
				_ = 0.5 * v * (1 + erff(v*(1/math.Sqrt2)))
			}
		}
	})
}
