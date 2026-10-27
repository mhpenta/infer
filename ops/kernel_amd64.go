//go:build amd64 && !purego

package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"golang.org/x/sys/cpu"
)

var (
	// hasAVX2 and hasAVX512 say what the CPU (and OS) support.
	hasAVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasAVX512 = hasAVX2 && cpu.X86.HasAVX512F

	// useAVX2 selects the 6×16 assembly kernel and the AVX2 vector loops;
	// useAVX512 the 12×32 kernel for full wide tiles.
	useAVX2   = hasAVX2
	useAVX512 bool

	// avx512Check is the result of the AVX-512 self-check at startup.
	avx512Check error
)

// init runs the AVX-512 kernel against the portable one before trusting it,
// since it cannot be tested on machines without AVX-512. Any mismatch keeps
// the AVX2 kernel; KernelCheck reports why.
func init() {
	if !hasAVX512 {
		return
	}
	if avx512Check = checkWideKernel(); avx512Check == nil {
		useAVX512, wide = true, true
	}
}

// checkWideKernel compares kernel12x32AVX512 with kernel12x32Portable on
// random data across depths around the unrolling and every epilogue mode.
func checkWideKernel() error {
	r := rand.New(rand.NewPCG(31, 32))
	const ldc = 2*nr + 3
	for _, k := range []int{1, 2, 3, 4, 7, 256, 301} {
		a0, a1 := randomFloats(r, k*mr), randomFloats(r, k*mr)
		b0, b1 := randomFloats(r, k*nr), randomFloats(r, k*nr)
		bias := randomFloats(r, 2*nr)
		start := randomFloats(r, 2*mr*ldc)
		h0, h1 := randomHalves(r, k*nr), randomHalves(r, k*nr)
		f0, f1 := randomBF16(r, k*nr), randomBF16(r, k*nr)
		q0, q1 := randomInt8(r, k*nr), randomInt8(r, k*nr)
		for _, mode := range []int{storeC, storeBiasC, addC} {
			got, want := append([]float32(nil), start...), append([]float32(nil), start...)
			kernel12x32AVX512(&a0[0], &a1[0], &b0[0], &b1[0], k, &got[0], ldc, &bias[0], mode)
			kernel12x32Portable(&a0[0], &a1[0], &b0[0], &b1[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnough(got, want); err != nil {
				return fmt.Errorf("k=%d mode=%d: %w", k, mode, err)
			}
			got, want = append(got[:0], start...), append(want[:0], start...)
			kernel12x32AVX512H(&a0[0], &a1[0], &h0[0], &h1[0], k, &got[0], ldc, &bias[0], mode)
			kernel12x32PortableNarrow(formatF16, &a0[0], &a1[0], &h0[0], &h1[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnough(got, want); err != nil {
				return fmt.Errorf("fp16 panels, k=%d mode=%d: %w", k, mode, err)
			}
			got, want = append(got[:0], start...), append(want[:0], start...)
			kernel12x32AVX512B(&a0[0], &a1[0], &f0[0], &f1[0], k, &got[0], ldc, &bias[0], mode)
			kernel12x32PortableNarrow(formatBF16, &a0[0], &a1[0], &f0[0], &f1[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnough(got, want); err != nil {
				return fmt.Errorf("bf16 panels, k=%d mode=%d: %w", k, mode, err)
			}
			got, want = append(got[:0], start...), append(want[:0], start...)
			kernel12x32AVX512I(&a0[0], &a1[0], &q0[0], &q1[0], k, &got[0], ldc, &bias[0], mode)
			kernel12x32PortableInt8(&a0[0], &a1[0], &q0[0], &q1[0], k, &want[0], ldc, &bias[0], mode)
			if err := closeEnough(got, want); err != nil {
				return fmt.Errorf("int8 panels, k=%d mode=%d: %w", k, mode, err)
			}
		}
	}
	return nil
}

// closeEnough compares a kernel's output with the portable kernel's.
func closeEnough(got, want []float32) error {
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); !(d <= 1e-4*(1+math.Abs(float64(want[i])))) {
			return fmt.Errorf("c[%d] = %g, portable %g", i, got[i], want[i])
		}
	}
	return nil
}

// Kernel names the microkernel in use, for logs.
func Kernel() string {
	switch {
	case useAVX512:
		return "avx512"
	case useAVX2:
		return "avx2"
	}
	return "go"
}

// KernelCheck returns why a SIMD kernel the CPU supports is not in use: a
// failed startup self-check. It is nil otherwise.
func KernelCheck() error {
	if hasAVX512 && avx512Check != nil {
		return fmt.Errorf("avx512 self-check failed: %w", avx512Check)
	}
	return nil
}

// UseKernel selects a kernel by name ("avx512", "avx2" or "go"), for
// comparing them on one machine. Call it at startup, before any model runs;
// it is not synchronized. It fails for a kernel this CPU cannot run or that
// failed its self-check.
func UseKernel(name string) error {
	switch name {
	case "avx512":
		if !hasAVX512 || avx512Check != nil {
			return fmt.Errorf("ops: avx512 kernel unavailable (cpu avx512f=%v, self-check: %v)", hasAVX512, avx512Check)
		}
		useAVX2, useAVX512, wide = true, true, true
	case "avx2":
		if !hasAVX2 {
			return fmt.Errorf("ops: avx2 kernel unavailable on this cpu")
		}
		useAVX2, useAVX512, wide = true, false, false
	case "go":
		useAVX2, useAVX512, wide = false, false, false
	default:
		return fmt.Errorf("ops: unknown kernel %q (want avx512, avx2 or go)", name)
	}
	return nil
}

// CPUFeatures lists the vector extensions the CPU reports, for logs.
func CPUFeatures() string {
	var have []string
	for _, f := range []struct {
		name string
		ok   bool
	}{
		{"avx2", cpu.X86.HasAVX2},
		{"fma", cpu.X86.HasFMA},
		{"avx512f", cpu.X86.HasAVX512F},
		{"avx512dq", cpu.X86.HasAVX512DQ},
		{"avx512bw", cpu.X86.HasAVX512BW},
		{"avx512vl", cpu.X86.HasAVX512VL},
		{"avx512vnni", cpu.X86.HasAVX512VNNI},
		{"avx512bf16", cpu.X86.HasAVX512BF16},
		{"amxtile", cpu.X86.HasAMXTile},
	} {
		if f.ok {
			have = append(have, f.name)
		}
	}
	return strings.Join(have, ",")
}

// kernel6x16AVX2 is the microkernel contract in AVX2/FMA assembly, epilogue
// included. The 6×16 block uses 12 of the 16 YMM registers as accumulators,
// two for the panel row and two for broadcasts.
//
//go:noescape
func kernel6x16AVX2(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int)

// kernel12x32AVX512 is the wide contract in AVX-512 assembly: 24 of the 32
// ZMM registers are accumulators, two hold the panel rows and six the
// broadcasts.
//
//go:noescape
func kernel12x32AVX512(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int)

// kernel6x16AVX2H is kernel6x16AVX2 for fp16 panels (F16C conversion; every
// AVX2+FMA CPU has F16C).
//
//go:noescape
func kernel6x16AVX2H(a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int)

// kernel12x32AVX512H is kernel12x32AVX512 for fp16 panels.
//
//go:noescape
func kernel12x32AVX512H(a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int)

// kernel6x16AVX2B is kernel6x16AVX2 for bf16 panels (widened with
// VPMOVZXWD and a 16-bit left shift).
//
//go:noescape
func kernel6x16AVX2B(a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int)

// kernel12x32AVX512B is kernel12x32AVX512 for bf16 panels.
//
//go:noescape
func kernel12x32AVX512B(a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int)

// HalfPanels reports whether fp16 panels pay off here: when the AVX2 kernels
// (which widen them in registers) are in use.
func HalfPanels() bool { return useAVX2 }

// BF16Panels reports whether bf16 panels run at SIMD speed here: when the
// AVX2 kernels (which widen them in registers) are in use.
func BF16Panels() bool { return useAVX2 }

// kernel6x16Narrow is the 16-bit-panel microkernel; see kernel_go.go.
func kernel6x16Narrow(f panelFormat, a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	switch {
	case !useAVX2:
		kernel6x16PortableNarrow(f, a, b, k, c, ldc, bias, mode)
	case f == formatBF16:
		kernel6x16AVX2B(a, b, k, c, ldc, bias, mode)
	default:
		kernel6x16AVX2H(a, b, k, c, ldc, bias, mode)
	}
}

// kernel12x32Narrow is the 16-bit-panel wide kernel; see kernel_go.go.
func kernel12x32Narrow(f panelFormat, a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	switch {
	case !useAVX512:
		kernel12x32PortableNarrow(f, a0, a1, b0, b1, k, c, ldc, bias, mode)
	case f == formatBF16:
		kernel12x32AVX512B(a0, a1, b0, b1, k, c, ldc, bias, mode)
	default:
		kernel12x32AVX512H(a0, a1, b0, b1, k, c, ldc, bias, mode)
	}
}

// kernel6x16 is the microkernel; see the contract in kernel_go.go.
func kernel6x16(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	if useAVX2 {
		kernel6x16AVX2(a, b, k, c, ldc, bias, mode)
		return
	}
	kernel6x16Portable(a, b, k, c, ldc, bias, mode)
}

// kernel12x32 is the wide kernel; see the contract in kernel_go.go.
func kernel12x32(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	if useAVX512 {
		kernel12x32AVX512(a0, a1, b0, b1, k, c, ldc, bias, mode)
		return
	}
	kernel12x32Portable(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

//go:noescape
func peakAVX2(iters int)

// peakLoop runs iters iterations of peakFlopsPerIter register-only flops and
// reports whether this build has such a loop.
func peakLoop(iters int) bool {
	if !useAVX2 {
		return false
	}
	peakAVX2(iters)
	return true
}

//go:noescape
func peakAVX512(iters int)

// peakWideLoop runs iters iterations of peakWideFlopsPerIter register-only
// flops at AVX-512 width and reports whether the CPU has AVX-512.
func peakWideLoop(iters int) bool {
	if !hasAVX512 {
		return false
	}
	peakAVX512(iters)
	return true
}

// One-row kernels (gemv_amd64.s): 4 or 1 panels, float32, fp16 or bf16
// panels (int8 ones are declared with the int8 dispatch below).
//
//go:noescape
func gemv4AVX2(a, b *float32, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1AVX2(a, b *float32, panelStride, k int, out *[nr]float32)

//go:noescape
func gemv4AVX2H(a *float32, b *uint16, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1AVX2H(a *float32, b *uint16, panelStride, k int, out *[nr]float32)

//go:noescape
func gemv4AVX2B(a *float32, b *uint16, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1AVX2B(a *float32, b *uint16, panelStride, k int, out *[nr]float32)

// gemvPanels computes one row against panels (1 or 4) consecutive panels;
// see gemvGo.
func gemvPanels(a, b *float32, panelStride, k, panels int, out *[4 * nr]float32) {
	switch {
	case !useAVX2:
		gemvGo(a, b, panelStride, k, panels, out[:])
	case panels == 4:
		gemv4AVX2(a, b, panelStride, k, out)
	default:
		gemv1AVX2(a, b, panelStride, k, (*[nr]float32)(out[:nr]))
	}
}

// gemvPanelsNarrow is gemvPanels for 16-bit panels in format f.
func gemvPanelsNarrow(f panelFormat, a *float32, b *uint16, panelStride, k, panels int, out *[4 * nr]float32) {
	one := (*[nr]float32)(out[:nr])
	switch {
	case !useAVX2:
		gemvGoNarrow(f, a, b, panelStride, k, panels, out[:])
	case f == formatBF16 && panels == 4:
		gemv4AVX2B(a, b, panelStride, k, out)
	case f == formatBF16:
		gemv1AVX2B(a, b, panelStride, k, one)
	case panels == 4:
		gemv4AVX2H(a, b, panelStride, k, out)
	default:
		gemv1AVX2H(a, b, panelStride, k, one)
	}
}

// kernel6x16AVX2I is kernel6x16AVX2 for int8 panels (widened with VPMOVSXBD
// and VCVTDQ2PS).
//
//go:noescape
func kernel6x16AVX2I(a *float32, b *int8, k int, c *float32, ldc int, bias *float32, mode int)

// kernel12x32AVX512I is kernel12x32AVX512 for int8 panels.
//
//go:noescape
func kernel12x32AVX512I(a0, a1 *float32, b0, b1 *int8, k int, c *float32, ldc int, bias *float32, mode int)

//go:noescape
func gemv4AVX2I(a *float32, b *int8, panelStride, k int, out *[4 * nr]float32)

//go:noescape
func gemv1AVX2I(a *float32, b *int8, panelStride, k int, out *[nr]float32)

// Int8Panels reports whether int8 panels run at SIMD speed here: when the
// AVX2 kernels (which widen them in registers) are in use.
func Int8Panels() bool { return useAVX2 }

// kernel6x16Int8 is the int8-panel microkernel; see kernel_go.go.
func kernel6x16Int8(a *float32, b *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	if useAVX2 {
		kernel6x16AVX2I(a, b, k, c, ldc, bias, mode)
		return
	}
	kernel6x16PortableInt8(a, b, k, c, ldc, bias, mode)
}

// kernel12x32Int8 is the int8-panel wide kernel; see kernel_go.go.
func kernel12x32Int8(a0, a1 *float32, b0, b1 *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	if useAVX512 {
		kernel12x32AVX512I(a0, a1, b0, b1, k, c, ldc, bias, mode)
		return
	}
	kernel12x32PortableInt8(a0, a1, b0, b1, k, c, ldc, bias, mode)
}

// gemvPanelsInt8 is gemvPanels for int8 panels.
func gemvPanelsInt8(a *float32, b *int8, panelStride, k, panels int, out *[4 * nr]float32) {
	switch {
	case !useAVX2:
		gemvGoInt8(a, b, panelStride, k, panels, out[:])
	case panels == 4:
		gemv4AVX2I(a, b, panelStride, k, out)
	default:
		gemv1AVX2I(a, b, panelStride, k, (*[nr]float32)(out[:nr]))
	}
}
