// Package quant compresses embeddings for storage and fast search.
package quant

// Binary quantizes v to one bit per dimension, matching sentence-transformers'
// quantize_embeddings(precision="binary"): bit = v[i] > 0, packed eight to a
// byte with the first dimension in the most significant bit (numpy's
// packbits), then shifted by -128 into an int8. len(v) must be a multiple of 8.
func Binary(v []float32) []int8 {
	if len(v)%8 != 0 {
		panic("quant.Binary: length must be a multiple of 8")
	}
	out := make([]int8, len(v)/8)
	for i := range out {
		var b byte
		for j, x := range v[i*8 : i*8+8] {
			if x > 0 {
				b |= 0x80 >> j
			}
		}
		out[i] = int8(int(b) - 128)
	}
	return out
}

// HammingBits returns how many bits differ between two Binary outputs.
func HammingBits(a, b []int8) int {
	n := 0
	for i := range a {
		x := byte(a[i]) ^ byte(b[i])
		for ; x != 0; x &= x - 1 {
			n++
		}
	}
	return n
}
