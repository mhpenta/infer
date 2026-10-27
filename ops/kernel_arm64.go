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

// BF16Panels reports whether bf16 panels run at SIMD speed here: true, as the
// NEON kernels widen them in registers.
func BF16Panels() bool { return true }

// kernel6x16NEONB is kernel6x16NEON for a bf16 panel (kernel_bf16_arm64.s).
//
//go:noescape
func kernel6x16NEONB(a *float32, b *uint16, k int, out *[mr * nr]float32)

// kernel6x16Narrow is the 16-bit-panel microkernel; see kernel_go.go. fp16
// panels only meet the portable kernel here, in tests.
func kernel6x16Narrow(f panelFormat, a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	if f != formatBF16 {
		kernel6x16PortableNarrow(f, a, b, k, c, ldc, bias, mode)
		return
	}
	var out [mr * nr]float32
	kernel6x16NEONB(a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

// kernel12x32Narrow is the 16-bit-panel wide kernel; only tests use it here.
func kernel12x32Narrow(f panelFormat, a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32PortableNarrow(f, a0, a1, b0, b1, k, c, ldc, bias, mode)
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

// One-row kernels for bf16 panels (kernel_bf16_arm64.s).
//
//go:noescape
func gemv4NEONB(a *float32, b *uint16, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1NEONB(a *float32, b *uint16, panelStride, k int, out *[nr]float32)

// gemvPanelsNarrow is gemvPanels for 16-bit panels in format f. fp16 panels
// only meet the portable loop here, in tests (HalfPanels is false).
func gemvPanelsNarrow(f panelFormat, a *float32, b *uint16, panelStride, k, panels int, out *[4 * nr]float32) {
	switch {
	case f != formatBF16:
		gemvGoNarrow(f, a, b, panelStride, k, panels, out[:])
	case panels == 4:
		gemv4NEONB(a, b, panelStride, k, out)
	default:
		gemv1NEONB(a, b, panelStride, k, (*[nr]float32)(out[:nr]))
	}
}

// Int8Panels reports whether int8 panels run at SIMD speed here: true, as
// the NEON kernels widen them in registers.
func Int8Panels() bool { return true }

// kernel6x16NEONI is kernel6x16NEON for an int8 panel (kernel_int8_arm64.s).
//
//go:noescape
func kernel6x16NEONI(a *float32, b *int8, k int, out *[mr * nr]float32)

// kernel6x16Int8 is the int8-panel microkernel; see kernel_go.go.
func kernel6x16Int8(a *float32, b *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	var out [mr * nr]float32
	kernel6x16NEONI(a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

// kernel12x32Int8 is the int8-panel wide kernel; only tests use it here.
func kernel12x32Int8(a0, a1 *float32, b0, b1 *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32PortableInt8(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// One-row kernels for int8 panels (kernel_int8_arm64.s).
//
//go:noescape
func gemv4NEONI(a *float32, b *int8, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1NEONI(a *float32, b *int8, panelStride, k int, out *[nr]float32)

// gemvPanelsInt8 is gemvPanels for int8 panels.
func gemvPanelsInt8(a *float32, b *int8, panelStride, k, panels int, out *[4 * nr]float32) {
	if panels == 4 {
		gemv4NEONI(a, b, panelStride, k, out)
		return
	}
	gemv1NEONI(a, b, panelStride, k, (*[nr]float32)(out[:nr]))
}
