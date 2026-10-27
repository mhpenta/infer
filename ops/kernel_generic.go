//go:build (!arm64 && !amd64) || purego

package ops

import "fmt"

// Kernel names the microkernel in use, for logs.
func Kernel() string { return "go" }

// CPUFeatures lists the vector extensions in use, for logs: none here.
func CPUFeatures() string { return "" }

// KernelCheck returns nil: there is no SIMD kernel to self-check.
func KernelCheck() error { return nil }

// UseKernel selects a kernel by name; only "go" exists in this build.
func UseKernel(name string) error {
	if name != "go" {
		return fmt.Errorf("ops: kernel %q unavailable in this build (only go)", name)
	}
	return nil
}

// kernel6x16 is the microkernel; see the contract in kernel_go.go.
func kernel6x16(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel6x16Portable(a, b, k, c, ldc, bias, mode)
}

// peakLoop reports that the portable build has no peak-rate loop.
func peakLoop(iters int) bool { return false }

// kernel12x32 is the wide kernel; see the contract in kernel_go.go. Only
// tests enable the wide path in this build.
func kernel12x32(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32Portable(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// peakWideLoop reports that this build has no wider vector unit to measure.
func peakWideLoop(iters int) bool { return false }

// HalfPanels reports whether fp16 panels pay off here: false, as there is no
// SIMD kernel that widens them (the portable one converts every value).
func HalfPanels() bool { return false }

// BF16Panels reports whether bf16 panels run at SIMD speed here: false, as
// the portable kernel converts every value.
func BF16Panels() bool { return false }

// kernel6x16Narrow is the 16-bit-panel microkernel; see kernel_go.go.
func kernel6x16Narrow(f panelFormat, a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel6x16PortableNarrow(f, a, b, k, c, ldc, bias, mode)
}

// kernel12x32Narrow is the 16-bit-panel wide kernel; only tests use it here.
func kernel12x32Narrow(f panelFormat, a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32PortableNarrow(f, a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// gemvPanels computes one row against panels (1 or 4) consecutive panels;
// see gemvGo.
func gemvPanels(a, b *float32, panelStride, k, panels int, out *[4 * nr]float32) {
	gemvGo(a, b, panelStride, k, panels, out[:])
}

// gemvPanelsNarrow is gemvPanels for 16-bit panels in format f.
func gemvPanelsNarrow(f panelFormat, a *float32, b *uint16, panelStride, k, panels int, out *[4 * nr]float32) {
	gemvGoNarrow(f, a, b, panelStride, k, panels, out[:])
}

// Int8Panels reports whether int8 panels run at SIMD speed here: false, as
// the portable kernel converts every value.
func Int8Panels() bool { return false }

// kernel6x16Int8 is the int8-panel microkernel; see kernel_go.go.
func kernel6x16Int8(a *float32, b *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel6x16PortableInt8(a, b, k, c, ldc, bias, mode)
}

// kernel12x32Int8 is the int8-panel wide kernel; only tests use it here.
func kernel12x32Int8(a0, a1 *float32, b0, b1 *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	kernel12x32PortableInt8(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// gemvPanelsInt8 is gemvPanels for int8 panels.
func gemvPanelsInt8(a *float32, b *int8, panelStride, k, panels int, out *[4 * nr]float32) {
	gemvGoInt8(a, b, panelStride, k, panels, out[:])
}
