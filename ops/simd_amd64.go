//go:build amd64 && !purego

package ops

import "math"

// AVX2 versions of the softmax and GELU loops (simd_amd64.s). Each wrapper
// runs the assembly on the largest multiple of 8 elements. scaleSlice runs
// the portable loop on the rest, which gives the same bits (one multiply per
// element); geluSlice does too, within a few ulps of the assembly; the
// softmax steps pad the rest to a block of 8 (below).

//go:noescape
func maxAVX2(x *float32, n int, out *[8]float32)

//go:noescape
func expShiftScaleAVX2(x *float32, n int, shift, scale float32, sums *[8]float32)

//go:noescape
func scaleAVX2(x *float32, n int, s float32)

//go:noescape
func geluAVX2(x *float32, n int)

// The softmax steps, maxSlice and expShiftScale, run the leftover entries
// past the last multiple of 8 through the same assembly, as one more block
// padded with -Inf, instead of through the portable loop. Entry j then always
// goes through lane j%8 of the vector code, whatever the row's length. A
// causal attention row is padded with -Inf up to the number of keys in its
// pass, so this is what makes a row's probabilities the same bits whether its
// pass was short (a Session's reused prefix) or long (the whole prompt): the
// assembly's exp differs from expf in the last bits, and the lane sums add in
// a different order than the portable loop. -Inf gives exactly 0 in the exp
// step and adds +0 to a lane, which changes nothing.

var negInfBlock = func() (b [8]float32) {
	for i := range b {
		b[i] = float32(math.Inf(-1))
	}
	return b
}()

// tailBlock copies the entries of x past main into an 8-entry block padded
// with -Inf.
func tailBlock(x []float32, main int) [8]float32 {
	block := negInfBlock
	copy(block[:], x[main:])
	return block
}

func maxSlice(x []float32) float32 {
	if !useAVX2 || len(x) == 0 {
		return maxGo(x)
	}
	main := len(x) &^ 7
	var lanes [8]float32
	if main > 0 {
		maxAVX2(&x[0], main, &lanes)
	}
	if main < len(x) {
		block := tailBlock(x, main)
		var tail [8]float32
		maxAVX2(&block[0], 8, &tail)
		if main == 0 {
			lanes = tail
		} else {
			for i := range lanes {
				lanes[i] = max(lanes[i], tail[i])
			}
		}
	}
	return maxGo(lanes[:])
}

func expShiftScale(x []float32, shift, scale float32) float64 {
	if !useAVX2 || len(x) == 0 {
		return expShiftScaleGo(x, shift, scale)
	}
	main := len(x) &^ 7
	var lanes [8]float32
	if main > 0 {
		expShiftScaleAVX2(&x[0], main, shift, scale, &lanes)
	}
	if main < len(x) {
		block := tailBlock(x, main)
		var tail [8]float32
		expShiftScaleAVX2(&block[0], 8, shift, scale, &tail)
		copy(x[main:], block[:])
		// Each tail lane holds one entry's exp (0 + e is exact), so adding it
		// to the lane's running sum is the same float32 addition the
		// assembly's loop would have done had the block been part of x.
		for i := range lanes {
			lanes[i] += tail[i]
		}
	}
	var sum float64
	for _, v := range lanes {
		sum += float64(v)
	}
	return sum
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
