package ops

import "math"

// Float32 versions of exp and erf for the activation and softmax loops. The
// float64 math package spends most of its time on precision float32 results
// cannot hold; these stay within a few float32 ulps (see mathf_test.go).

// expf returns e**x. It reduces x = n·ln2 + r with |r| ≤ ln2/2, evaluates a
// degree-5 polynomial for e**r (the Cephes expf coefficients) and scales by
// 2**n through the exponent bits.
func expf(x float32) float32 {
	const (
		maxArg = 88.72283905206835  // above this e**x overflows float32
		minArg = -87.33654475055310 // below this e**x is subnormal; flush to 0
		log2e  = 1.44269504088896341
		ln2Hi  = 0.693359375 // ln2 split so n·ln2Hi is exact
		ln2Lo  = -2.12194440e-4
	)
	switch {
	case x > maxArg:
		return float32(math.Inf(1))
	case x < minArg:
		return 0 // also covers -Inf, used for masked positions
	case x != x:
		return x // NaN
	}
	// n = round(x·log2e), computed as floor(t + 0.5).
	t := x*log2e + 0.5
	n := int32(t)
	if t < float32(n) {
		n--
	}
	fn := float32(n)
	r := x - fn*ln2Hi - fn*ln2Lo
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	y := p*r*r + r + 1
	if n > 127 {
		// 2**128 has no float32 encoding, but the product can still be finite.
		return y * 2 * math.Float32frombits(uint32(n-1+127)<<23)
	}
	return y * math.Float32frombits(uint32(n+127)<<23)
}

// erff returns the error function of x as a ratio of an odd and an even
// polynomial in x (the minimax approximation used by Eigen and TensorFlow).
// Inputs are clamped to [-4, 4], beyond which erf is ±1 in float32.
func erff(x float32) float32 {
	x = min(max(x, -4), 4)
	x2 := x * x
	p := float32(-2.72614225801306e-10)
	p = p*x2 + 2.77068142495902e-08
	p = p*x2 - 2.10102402082508e-06
	p = p*x2 - 5.69250639462346e-05
	p = p*x2 - 7.34990630326855e-04
	p = p*x2 - 2.95459980854025e-03
	p = p*x2 - 1.60960333262415e-02
	p *= x
	q := float32(-1.45660718464996e-05)
	q = q*x2 - 2.13374055278905e-04
	q = q*x2 - 1.68282697438203e-03
	q = q*x2 - 7.37332916720468e-03
	q = q*x2 - 1.42647390514189e-02
	return p / q
}

// Portable slice loops behind softmax and GELU. On amd64 with AVX2 the
// dispatchers in simd_amd64.go run 8 lanes at a time and use these only for
// the leftover tail, so both paths compute the same formulas.

// maxGo returns the largest element of a non-empty x.
func maxGo(x []float32) float32 {
	peak := x[0]
	for _, v := range x[1:] {
		peak = max(peak, v)
	}
	return peak
}

// expShiftScaleGo replaces each x[i] with e**((x[i]-shift)·scale) and returns
// the sum of the results.
func expShiftScaleGo(x []float32, shift, scale float32) float64 {
	var sum float64
	for i, v := range x {
		e := expf((v - shift) * scale)
		x[i] = e
		sum += float64(e)
	}
	return sum
}

// scaleGo multiplies each element of x by s.
func scaleGo(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}

// geluGo applies GELU in place.
func geluGo(x []float32) {
	for i, v := range x {
		x[i] = 0.5 * v * (1 + erff(v*(1/math.Sqrt2)))
	}
}
