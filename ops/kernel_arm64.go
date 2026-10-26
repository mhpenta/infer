//go:build arm64 && !purego

package ops

import "fmt"

// Kernel names the microkernel in use, for logs.
func Kernel() string { return "neon" }

// CPUFeatures lists the vector extensions in use, for logs.
func CPUFeatures() string { return "neon" }

// KernelCheck returns nil: the NEON kernel needs no self-check (every arm64
// CPU has NEON, and the tests run it).
func KernelCheck() error { return nil }

// UseKernel selects a kernel by name; only "neon" exists in this build.
func UseKernel(name string) error {
	if name != "neon" {
		return fmt.Errorf("ops: kernel %q unavailable in this build (only neon)", name)
	}
	return nil
}

// kernel6x16NEON is kernel6x16Go in NEON assembly, k >= 1. The 6×16 block
// uses 24 of the 32 vector registers as accumulators, four for the panel row
// and four for broadcasts.
//
//go:noescape
func kernel6x16NEON(a, b *float32, k int, out *[mr * nr]float32)

// kernel6x16 is the microkernel; see the contract in kernel_go.go. On arm64
// the epilogue stays in Go: production runs on amd64, where it is fused.
func kernel6x16(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	var out [mr * nr]float32
	kernel6x16NEON(a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

//go:noescape
func peakNEON(iters int)

// peakLoop runs iters iterations of peakFlopsPerIter register-only flops and
// reports whether this build has such a loop.
func peakLoop(iters int) bool {
	peakNEON(iters)
	return true
}

// kernel12x32 is the wide kernel; see the contract in kernel_go.go. arm64 has
// no wide kernel (NEON has 32 registers of 4 lanes, which 6×16 already uses),
// so only tests enable the wide path here.
func kernel12x32(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32Portable(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// peakWideLoop reports that this build has no wider vector unit to measure.
func peakWideLoop(iters int) bool { return false }

// HalfPanels reports whether fp16 panels pay off here: false, as there is no
// SIMD kernel that widens them (the portable one converts every value).
func HalfPanels() bool { return false }

// kernel6x16Half is the fp16-panel microkernel; only tests use it here.
func kernel6x16Half(a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel6x16PortableHalf(a, b, k, c, ldc, bias, mode)
}

// kernel12x32Half is the fp16-panel wide kernel; only tests use it here.
func kernel12x32Half(a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32PortableHalf(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// One-row kernels (gemv_arm64.s), 4 or 1 float32 panels.
//
//go:noescape
func gemv4NEON(a, b *float32, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1NEON(a, b *float32, panelStride, k int, out *[nr]float32)

// gemvPanels computes one row against panels (1 or 4) consecutive panels;
// see gemvGo.
func gemvPanels(a, b *float32, panelStride, k, panels int, out *[4 * nr]float32) {
	if panels == 4 {
		gemv4NEON(a, b, panelStride, k, out)
		return
	}
	gemv1NEON(a, b, panelStride, k, (*[nr]float32)(out[:nr]))
}

// gemvPanelsHalf is gemvPanels for fp16 panels, which arm64 only meets in
// tests (HalfPanels is false here).
func gemvPanelsHalf(a *float32, b *uint16, panelStride, k, panels int, out *[4 * nr]float32) {
	gemvGoHalf(a, b, panelStride, k, panels, out[:])
}
