// Package half converts between IEEE 754 half precision (fp16, stored as
// uint16 bits) and float32.
package half

import "math"

// ToFloat32 converts fp16 bits to float32 exactly.
func ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: shift until the implicit leading bit appears.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13) // Inf or NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// FromFloat32Exact returns the fp16 bits of f and true when f is exactly
// representable in fp16, or false when converting would round. NaN is never
// exact.
func FromFloat32Exact(f float32) (uint16, bool) {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int(bits>>23) & 0xff
	mant := bits & 0x7fffff
	switch {
	case exp == 0 && mant == 0:
		return sign, true // ±0
	case exp == 0xff:
		if mant != 0 {
			return 0, false // NaN
		}
		return sign | 0x7c00, true // ±Inf
	}
	e := exp - 127 + 15
	switch {
	case e >= 0x1f:
		return 0, false // too large
	case e >= 1:
		if mant&0x1fff != 0 {
			return 0, false // more than 10 mantissa bits
		}
		return sign | uint16(e)<<10 | uint16(mant>>13), true
	case e >= -10:
		// fp16 subnormal: value = (1.mant) · 2^(exp-127) = m · 2^-24.
		full := mant | 0x800000
		shift := uint(14 - e)
		if full&(1<<shift-1) != 0 {
			return 0, false
		}
		return sign | uint16(full>>shift), true
	}
	return 0, false // below the smallest subnormal
}
