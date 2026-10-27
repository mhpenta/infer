package ops

import (
	"math"
	"unsafe"

	"github.com/mhpenta/infer/internal/half"
)

// The microkernel contract, shared by every implementation:
//
//	kernel6x16(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int)
//
// a is a packed strip (k steps of mr values, see packA), b a packed panel (k
// steps of nr values), k >= 1. The mr×nr sums of outer products are written
// to c, whose rows are ldc elements apart, according to mode: storeC,
// storeBiasC (bias points at nr values) or addC.

// kernel6x16Go computes the mr×nr sums into out (row-major). It is the
// portable kernel, used where no SIMD kernel exists or with the purego build
// tag, and accumulates in the same order along k as the SIMD kernels.
func kernel6x16Go(a, b *float32, k int, out *[mr * nr]float32) {
	strip := unsafe.Slice(a, k*mr)
	panel := unsafe.Slice(b, k*nr)
	var acc [mr * nr]float32
	for p := range k {
		bp := panel[p*nr : p*nr+nr : p*nr+nr]
		ap := strip[p*mr : p*mr+mr : p*mr+mr]
		for r, av := range ap {
			ar := acc[r*nr : r*nr+nr : r*nr+nr]
			for j, bv := range bp {
				ar[j] += av * bv
			}
		}
	}
	*out = acc
}

// epilogueGo writes the sums in out to c according to mode.
func epilogueGo(out *[mr * nr]float32, c *float32, ldc int, bias *float32, mode int) {
	var b []float32
	if mode == storeBiasC {
		b = unsafe.Slice(bias, nr)
	}
	for r := range mr {
		dst := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(c), r*ldc*4)), nr)
		src := out[r*nr : r*nr+nr]
		switch mode {
		case storeC:
			copy(dst, src)
		case storeBiasC:
			for j, v := range src {
				dst[j] = v + b[j]
			}
		case addC:
			for j, v := range src {
				dst[j] += v
			}
		}
	}
}

// kernel6x16Portable is the whole microkernel contract in Go.
func kernel6x16Portable(a, b *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	var out [mr * nr]float32
	kernel6x16Go(a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

// The wide kernel contract, used for full 12×32 tiles when a wide SIMD kernel
// is active (see wide in matmul.go):
//
//	kernel12x32(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int)
//
// It is the 6×16 contract over two adjacent packed strips (a0 for rows 0-5,
// a1 for rows 6-11) and two adjacent panels (b0 for columns 0-15, b1 for
// 16-31), so it needs no packing of its own; bias points at 32 values.

// kernel12x32Portable is the wide contract as four 6×16 portable kernels.
func kernel12x32Portable(a0, a1, b0, b1 *float32, k int, c *float32, ldc int, bias *float32, mode int) {
	bias1 := bias
	if bias != nil {
		bias1 = (*float32)(unsafe.Add(unsafe.Pointer(bias), nr*4))
	}
	right := (*float32)(unsafe.Add(unsafe.Pointer(c), nr*4))
	down := (*float32)(unsafe.Add(unsafe.Pointer(c), mr*ldc*4))
	downRight := (*float32)(unsafe.Add(unsafe.Pointer(down), nr*4))
	kernel6x16Portable(a0, b0, k, c, ldc, bias, mode)
	kernel6x16Portable(a0, b1, k, right, ldc, bias1, mode)
	kernel6x16Portable(a1, b0, k, down, ldc, bias, mode)
	kernel6x16Portable(a1, b1, k, downRight, ldc, bias1, mode)
}

// The narrow-panel contracts are the same with b (and b0, b1) pointing at
// 16-bit panels (*uint16) in format f; each value is widened to float32
// before use.

// widenRow converts one panel row of 16-bit values to float32.
func widenRow(f panelFormat, dst *[nr]float32, src []uint16) {
	if f == formatBF16 {
		for j, h := range src[:nr] {
			dst[j] = math.Float32frombits(uint32(h) << 16)
		}
		return
	}
	for j, h := range src[:nr] {
		dst[j] = half.ToFloat32(h)
	}
}

// kernel6x16GoNarrow is kernel6x16Go for a 16-bit panel.
func kernel6x16GoNarrow(f panelFormat, a *float32, b *uint16, k int, out *[mr * nr]float32) {
	strip := unsafe.Slice(a, k*mr)
	panel := unsafe.Slice(b, k*nr)
	var acc [mr * nr]float32
	var bp [nr]float32
	for p := range k {
		widenRow(f, &bp, panel[p*nr:])
		ap := strip[p*mr : p*mr+mr : p*mr+mr]
		for r, av := range ap {
			ar := acc[r*nr : r*nr+nr : r*nr+nr]
			for j, bv := range bp {
				ar[j] += av * bv
			}
		}
	}
	*out = acc
}

// kernel6x16PortableNarrow is the narrow-panel microkernel contract in Go.
func kernel6x16PortableNarrow(f panelFormat, a *float32, b *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	var out [mr * nr]float32
	kernel6x16GoNarrow(f, a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

// kernel12x32PortableNarrow is the narrow-panel wide contract as four 6×16
// portable kernels.
func kernel12x32PortableNarrow(f panelFormat, a0, a1 *float32, b0, b1 *uint16, k int, c *float32, ldc int, bias *float32, mode int) {
	bias1 := bias
	if bias != nil {
		bias1 = (*float32)(unsafe.Add(unsafe.Pointer(bias), nr*4))
	}
	right := (*float32)(unsafe.Add(unsafe.Pointer(c), nr*4))
	down := (*float32)(unsafe.Add(unsafe.Pointer(c), mr*ldc*4))
	downRight := (*float32)(unsafe.Add(unsafe.Pointer(down), nr*4))
	kernel6x16PortableNarrow(f, a0, b0, k, c, ldc, bias, mode)
	kernel6x16PortableNarrow(f, a0, b1, k, right, ldc, bias1, mode)
	kernel6x16PortableNarrow(f, a1, b0, k, down, ldc, bias, mode)
	kernel6x16PortableNarrow(f, a1, b1, k, downRight, ldc, bias1, mode)
}

// gemvGo computes out[16p + j], the dot product of a[0:k] with column j of
// panel p, for panels consecutive panels starting at b (panelStride values
// apart). It sums in the same order as kernel6x16Go.
func gemvGo(a, b *float32, panelStride, k, panels int, out []float32) {
	as := unsafe.Slice(a, k)
	for p := range panels {
		panel := unsafe.Slice((*float32)(unsafe.Add(unsafe.Pointer(b), p*panelStride*4)), k*nr)
		var acc [nr]float32
		for kk, av := range as {
			bp := panel[kk*nr : kk*nr+nr : kk*nr+nr]
			for j, bv := range bp {
				acc[j] += av * bv
			}
		}
		copy(out[p*nr:], acc[:])
	}
}

// gemvGoNarrow is gemvGo for 16-bit panels in format f.
func gemvGoNarrow(f panelFormat, a *float32, b *uint16, panelStride, k, panels int, out []float32) {
	as := unsafe.Slice(a, k)
	for p := range panels {
		panel := unsafe.Slice((*uint16)(unsafe.Add(unsafe.Pointer(b), p*panelStride*2)), k*nr)
		var acc, bp [nr]float32
		for kk, av := range as {
			widenRow(f, &bp, panel[kk*nr:])
			for j, bv := range bp {
				acc[j] += av * bv
			}
		}
		copy(out[p*nr:], acc[:])
	}
}

// The int8-panel contracts are the same with b (and b0, b1) pointing at int8
// panels; each value is converted to float32 (exactly) before use, so the
// sums are those of float32 panels holding the integers. Scaling is left to
// the caller (see PackInt8).

// kernel6x16GoInt8 is kernel6x16Go for an int8 panel.
func kernel6x16GoInt8(a *float32, b *int8, k int, out *[mr * nr]float32) {
	strip := unsafe.Slice(a, k*mr)
	panel := unsafe.Slice(b, k*nr)
	var acc [mr * nr]float32
	var bp [nr]float32
	for p := range k {
		for j, q := range panel[p*nr : p*nr+nr] {
			bp[j] = float32(q)
		}
		ap := strip[p*mr : p*mr+mr : p*mr+mr]
		for r, av := range ap {
			ar := acc[r*nr : r*nr+nr : r*nr+nr]
			for j, bv := range bp {
				ar[j] += av * bv
			}
		}
	}
	*out = acc
}

// kernel6x16PortableInt8 is the int8-panel microkernel contract in Go.
func kernel6x16PortableInt8(a *float32, b *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	var out [mr * nr]float32
	kernel6x16GoInt8(a, b, k, &out)
	epilogueGo(&out, c, ldc, bias, mode)
}

// kernel12x32PortableInt8 is the int8-panel wide contract as four 6×16
// portable kernels.
func kernel12x32PortableInt8(a0, a1 *float32, b0, b1 *int8, k int, c *float32, ldc int, bias *float32, mode int) {
	bias1 := bias
	if bias != nil {
		bias1 = (*float32)(unsafe.Add(unsafe.Pointer(bias), nr*4))
	}
	right := (*float32)(unsafe.Add(unsafe.Pointer(c), nr*4))
	down := (*float32)(unsafe.Add(unsafe.Pointer(c), mr*ldc*4))
	downRight := (*float32)(unsafe.Add(unsafe.Pointer(down), nr*4))
	kernel6x16PortableInt8(a0, b0, k, c, ldc, bias, mode)
	kernel6x16PortableInt8(a0, b1, k, right, ldc, bias1, mode)
	kernel6x16PortableInt8(a1, b0, k, down, ldc, bias, mode)
	kernel6x16PortableInt8(a1, b1, k, downRight, ldc, bias1, mode)
}

// gemvGoInt8 is gemvGo for int8 panels.
func gemvGoInt8(a *float32, b *int8, panelStride, k, panels int, out []float32) {
	as := unsafe.Slice(a, k)
	for p := range panels {
		panel := unsafe.Slice((*int8)(unsafe.Add(unsafe.Pointer(b), p*panelStride)), k*nr)
		var acc [nr]float32
		for kk, av := range as {
			for j, q := range panel[kk*nr : kk*nr+nr] {
				acc[j] += av * float32(q)
			}
		}
		copy(out[p*nr:], acc[:])
	}
}
