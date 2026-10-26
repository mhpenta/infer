//go:build !amd64 || purego

package ops

func maxSlice(x []float32) float32 { return maxGo(x) }
func expShiftScale(x []float32, shift, scale float32) float64 {
	return expShiftScaleGo(x, shift, scale)
}
func scaleSlice(x []float32, s float32) { scaleGo(x, s) }
func geluSlice(x []float32)             { geluGo(x) }
