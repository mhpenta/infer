//go:build amd64 && !purego

package ops

// AVX2 versions of the softmax and GELU loops (simd_amd64.s). Each wrapper
// runs the assembly on the largest multiple of 8 elements and the portable
// loop on the rest.

//go:noescape
func maxAVX2(x *float32, n int, out *[8]float32)

//go:noescape
func expShiftScaleAVX2(x *float32, n int, shift, scale float32, sums *[8]float32)

//go:noescape
func scaleAVX2(x *float32, n int, s float32)

//go:noescape
func geluAVX2(x *float32, n int)

func maxSlice(x []float32) float32 {
	main := len(x) &^ 7
	if !useAVX2 || main == 0 {
		return maxGo(x)
	}
	var lanes [8]float32
	maxAVX2(&x[0], main, &lanes)
	peak := maxGo(lanes[:])
	if main < len(x) {
		peak = max(peak, maxGo(x[main:]))
	}
	return peak
}

func expShiftScale(x []float32, shift, scale float32) float64 {
	main := len(x) &^ 7
	if !useAVX2 || main == 0 {
		return expShiftScaleGo(x, shift, scale)
	}
	var lanes [8]float32
	expShiftScaleAVX2(&x[0], main, shift, scale, &lanes)
	var sum float64
	for _, v := range lanes {
		sum += float64(v)
	}
	return sum + expShiftScaleGo(x[main:], shift, scale)
}

func scaleSlice(x []float32, s float32) {
	main := len(x) &^ 7
	if !useAVX2 || main == 0 {
		scaleGo(x, s)
		return
	}
	scaleAVX2(&x[0], main, s)
	scaleGo(x[main:], s)
}

func geluSlice(x []float32) {
	main := len(x) &^ 7
	if !useAVX2 || main == 0 {
		geluGo(x)
		return
	}
	geluAVX2(&x[0], main)
	geluGo(x[main:])
}
