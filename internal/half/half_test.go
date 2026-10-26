package half

import (
	"math"
	"testing"
)

func TestRoundTripAllHalves(t *testing.T) {
	for h := range 1 << 16 {
		f := ToFloat32(uint16(h))
		if f != f {
			continue // NaN
		}
		got, ok := FromFloat32Exact(f)
		if !ok || got != uint16(h) {
			t.Fatalf("FromFloat32Exact(ToFloat32(%#04x) = %g) = %#04x, %v", h, f, got, ok)
		}
	}
}

func TestToFloat32(t *testing.T) {
	tests := []struct {
		in   uint16
		want float32
	}{
		{0x3c00, 1},
		{0xc000, -2},
		{0x7bff, 65504},
		{0x0001, float32(math.Ldexp(1, -24))},
		{0x03ff, float32(math.Ldexp(1023, -24))},
		{0x0400, float32(math.Ldexp(1, -14))},
		{0xfc00, float32(math.Inf(-1))},
	}
	for _, tt := range tests {
		if got := ToFloat32(tt.in); got != tt.want {
			t.Errorf("ToFloat32(%#04x) = %g, want %g", tt.in, got, tt.want)
		}
	}
}

func TestFromFloat32ExactRejectsRounding(t *testing.T) {
	for _, f := range []float32{0.1, 1 + 1.0/2048, 65520, float32(math.Ldexp(1, -25)), float32(math.NaN())} {
		if _, ok := FromFloat32Exact(f); ok {
			t.Errorf("FromFloat32Exact(%g) reported exact", f)
		}
	}
}
