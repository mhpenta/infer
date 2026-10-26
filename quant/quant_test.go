package quant

import (
	"slices"
	"testing"
)

func TestBinary(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
		want []int8
	}{
		// bits 10000000 = 128 → 0; 00000000 → -128; 11111111 = 255 → 127
		{"first dimension is the high bit", []float32{1, 0, -1, 0, 0, 0, 0, 0}, []int8{0}},
		{"zero is not positive", []float32{0, 0, 0, 0, 0, 0, 0, 0}, []int8{-128}},
		{"all positive", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []int8{127}},
		{"last dimension is the low bit", []float32{-1, -1, -1, -1, -1, -1, -1, 0.5}, []int8{-127}},
		{"two bytes", []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}, []int8{0, 64 - 128}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Binary(tt.in); !slices.Equal(got, tt.want) {
				t.Errorf("Binary(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestHammingBits(t *testing.T) {
	a := []int8{0, -128, 127}
	b := []int8{1, -128, -128} // 1 bit, 0 bits, 8 bits
	if got := HammingBits(a, b); got != 9 {
		t.Errorf("HammingBits = %d, want 9", got)
	}
}
