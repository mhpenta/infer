package ops

import (
	"math"
	"math/rand/v2"
	"time"

	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/tensor"
)

// Diagnostics for measuring the kernels where they run, such as inside a
// container on a cloud instance. Each runs single-threaded on the calling goroutine for
// about d and returns billions of floating-point operations per second.

// peakFlopsPerIter is the work of one peakLoop iteration on either SIMD
// architecture: 12 AVX2 FMAs × 8 lanes or 24 NEON FMAs × 4 lanes, × 2 flops.
const peakFlopsPerIter = 192

// peakWideFlopsPerIter is the work of one peakWideLoop iteration: 24 AVX-512
// FMAs × 16 lanes × 2 flops.
const peakWideFlopsPerIter = 768

// PeakWideGFLOPS is PeakGFLOPS at AVX-512 width, or 0 without AVX-512.
func PeakWideGFLOPS(d time.Duration) float64 {
	const iters = 1 << 12
	if !peakWideLoop(1) {
		return 0
	}
	return measure(d, func() { peakWideLoop(iters) }, iters*peakWideFlopsPerIter)
}

// PeakGFLOPS measures this core's fused multiply-add rate with independent
// register-only FMAs: the ceiling for any matmul kernel here. It returns 0 in
// builds without a SIMD kernel.
func PeakGFLOPS(d time.Duration) float64 {
	const iters = 1 << 14
	if !peakLoop(1) {
		return 0
	}
	return measure(d, func() { peakLoop(iters) }, iters*peakFlopsPerIter)
}

// KernelGFLOPS times each microkernel in use alone, kc deep, on operands
// that stay in L1: the best a full matmul could reach with it. Keys are the
// tile shapes, "6x16" and, when the wide kernel is active, "12x32".
func KernelGFLOPS(d time.Duration) map[string]float64 {
	r := rand.New(rand.NewPCG(1, 2))
	a0, a1 := randomFloats(r, kc*mr), randomFloats(r, kc*mr)
	b0, b1 := randomFloats(r, kc*nr), randomFloats(r, kc*nr)
	c := make([]float32, 4*mr*nr)
	rates := map[string]float64{
		"6x16": measure(d, func() { kernel6x16(&a0[0], &b0[0], kc, &c[0], nr, nil, storeC) }, 2*mr*nr*kc),
	}
	if wide {
		rates["12x32"] = measure(d, func() {
			kernel12x32(&a0[0], &a1[0], &b0[0], &b1[0], kc, &c[0], 2*nr, nil, storeC)
		}, 4*2*mr*nr*kc)
	}
	return rates
}

// MatMulGFLOPS times a single-threaded product of an [m, k] matrix with a
// packed [k, n] one, as Linear runs it, with "f32" or "f16" panels.
func MatMulGFLOPS(d time.Duration, m, n, k int, panels string) float64 {
	r := rand.New(rand.NewPCG(3, 4))
	a := randomFloats(r, m*k)
	wt := tensor.New(n, k)
	for i, h := range randomHalves(r, n*k) {
		wt.Data[i] = halfToFloat(h)
	}
	w := Pack(wt)
	switch panels {
	case "f32":
	case "f16":
		var err error
		if w, err = PackHalf(wt); err != nil {
			panic(err)
		}
	default:
		panic("ops.MatMulGFLOPS: unknown panels " + panels)
	}
	c := make([]float32, m*n)
	return measure(d, func() { gemm(m, a, k, w, nil, c, n, 1) }, float64(2*m*n*k))
}

// measure calls fn repeatedly for about d (at least once, after a warm-up
// call) and returns the rate in GFLOP/s for flops per call.
func measure(d time.Duration, fn func(), flops float64) float64 {
	fn()
	calls := 0
	start := time.Now()
	for calls == 0 || time.Since(start) < d {
		fn()
		calls++
	}
	return flops * float64(calls) / time.Since(start).Seconds() / 1e9
}

// randomHalves returns n random fp16 values (bits): normal draws on a 1/256
// grid, which fp16 holds exactly for |x| < 8.
func randomHalves(r *rand.Rand, n int) []uint16 {
	s := make([]uint16, n)
	for i := range s {
		v := float32(math.Round(min(max(r.NormFloat64(), -7), 7)*256)) / 256
		h, ok := half.FromFloat32Exact(v)
		if !ok {
			panic("randomHalves: value not exact in fp16")
		}
		s[i] = h
	}
	return s
}

func halfToFloat(h uint16) float32 { return half.ToFloat32(h) }

func randomFloats(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}
