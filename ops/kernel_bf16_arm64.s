//go:build arm64 && !purego

#include "textflag.h"

// NEON kernels for bf16 panels: kernel_arm64.s and gemv_arm64.s with each
// panel row widened from bf16 in registers.

// func kernel6x16NEONB(a *float32, b *uint16, k int, out *[96]float32)
//
// kernel6x16NEON for a bf16 panel. Per k step it loads one 16-wide panel row
// of bf16 values (V0, V1) and widens it into V2, V3, V0, V1 (columns 0-3,
// 4-7, 8-11, 12-15): ZIP1/ZIP2 with the zero register V4 interleave each
// 16-bit value above 16 zero bits, which is the float32 a bf16 value
// encodes. It then broadcasts the 6 values of the packed a strip (V5-V7,
// rotating) and issues 24 fused multiply-adds into V8-V31 in the same order
// as kernel6x16NEON, so results are bit-identical to its on the widened
// values.
TEXT ·kernel6x16NEONB(SB), NOSPLIT, $0-32
	MOVD a+0(FP), R1
	ADD  $4, R1, R2
	ADD  $8, R1, R3
	ADD  $12, R1, R4
	ADD  $16, R1, R5
	ADD  $20, R1, R6
	MOVD $24, R10 // strip stride: mr values
	MOVD b+8(FP), R7
	MOVD k+16(FP), R8
	MOVD out+24(FP), R9

	VEOR V4.B16, V4.B16, V4.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16
	VEOR V25.B16, V25.B16, V25.B16
	VEOR V26.B16, V26.B16, V26.B16
	VEOR V27.B16, V27.B16, V27.B16
	VEOR V28.B16, V28.B16, V28.B16
	VEOR V29.B16, V29.B16, V29.B16
	VEOR V30.B16, V30.B16, V30.B16
	VEOR V31.B16, V31.B16, V31.B16

loop:
	VLD1.P 32(R7), [V0.H8, V1.H8]
	VZIP1 V0.H8, V4.H8, V2.H8
	VZIP2 V0.H8, V4.H8, V3.H8
	VZIP1 V1.H8, V4.H8, V0.H8
	VZIP2 V1.H8, V4.H8, V1.H8

	VLD1R.P (R1)(R10), [V5.S4]
	VFMLA   V2.S4, V5.S4, V8.S4
	VFMLA   V3.S4, V5.S4, V9.S4
	VFMLA   V0.S4, V5.S4, V10.S4
	VFMLA   V1.S4, V5.S4, V11.S4

	VLD1R.P (R2)(R10), [V6.S4]
	VFMLA   V2.S4, V6.S4, V12.S4
	VFMLA   V3.S4, V6.S4, V13.S4
	VFMLA   V0.S4, V6.S4, V14.S4
	VFMLA   V1.S4, V6.S4, V15.S4

	VLD1R.P (R3)(R10), [V7.S4]
	VFMLA   V2.S4, V7.S4, V16.S4
	VFMLA   V3.S4, V7.S4, V17.S4
	VFMLA   V0.S4, V7.S4, V18.S4
	VFMLA   V1.S4, V7.S4, V19.S4

	VLD1R.P (R4)(R10), [V5.S4]
	VFMLA   V2.S4, V5.S4, V20.S4
	VFMLA   V3.S4, V5.S4, V21.S4
	VFMLA   V0.S4, V5.S4, V22.S4
	VFMLA   V1.S4, V5.S4, V23.S4

	VLD1R.P (R5)(R10), [V6.S4]
	VFMLA   V2.S4, V6.S4, V24.S4
	VFMLA   V3.S4, V6.S4, V25.S4
	VFMLA   V0.S4, V6.S4, V26.S4
	VFMLA   V1.S4, V6.S4, V27.S4

	VLD1R.P (R6)(R10), [V7.S4]
	VFMLA   V2.S4, V7.S4, V28.S4
	VFMLA   V3.S4, V7.S4, V29.S4
	VFMLA   V0.S4, V7.S4, V30.S4
	VFMLA   V1.S4, V7.S4, V31.S4

	SUBS $1, R8, R8
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R9)
	VST1.P [V12.S4, V13.S4, V14.S4, V15.S4], 64(R9)
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R9)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R9)
	VST1.P [V24.S4, V25.S4, V26.S4, V27.S4], 64(R9)
	VST1.P [V28.S4, V29.S4, V30.S4, V31.S4], 64(R9)
	RET

// func gemv4NEONB(a *float32, b *uint16, panelStride int, k int, out *[64]float32)
//
// gemv4NEON for bf16 panels: per k step, broadcast a[kk] (V0), load one
// 16-wide bf16 row of each panel, widen it with ZIP1/ZIP2 against the zero
// register V5 (as in kernel6x16NEONB) and fused multiply-add it into the
// panel's four accumulators, in the same order as gemv4NEON.
TEXT ·gemv4NEONB(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD panelStride+16(FP), R5
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7
	LSL  $1, R5 // panel stride in bytes
	ADD  R5, R1, R2
	ADD  R5, R2, R3
	ADD  R5, R3, R4

	VEOR V5.B16, V5.B16, V5.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

loop:
	VLD1R.P 4(R0), [V0.S4]
	VLD1.P 32(R1), [V24.H8, V25.H8]
	VZIP1 V24.H8, V5.H8, V26.H8
	VZIP2 V24.H8, V5.H8, V27.H8
	VZIP1 V25.H8, V5.H8, V28.H8
	VZIP2 V25.H8, V5.H8, V29.H8
	VFMLA V26.S4, V0.S4, V8.S4
	VFMLA V27.S4, V0.S4, V9.S4
	VFMLA V28.S4, V0.S4, V10.S4
	VFMLA V29.S4, V0.S4, V11.S4
	VLD1.P 32(R2), [V30.H8, V31.H8]
	VZIP1 V30.H8, V5.H8, V1.H8
	VZIP2 V30.H8, V5.H8, V2.H8
	VZIP1 V31.H8, V5.H8, V3.H8
	VZIP2 V31.H8, V5.H8, V4.H8
	VFMLA V1.S4, V0.S4, V12.S4
	VFMLA V2.S4, V0.S4, V13.S4
	VFMLA V3.S4, V0.S4, V14.S4
	VFMLA V4.S4, V0.S4, V15.S4
	VLD1.P 32(R3), [V24.H8, V25.H8]
	VZIP1 V24.H8, V5.H8, V26.H8
	VZIP2 V24.H8, V5.H8, V27.H8
	VZIP1 V25.H8, V5.H8, V28.H8
	VZIP2 V25.H8, V5.H8, V29.H8
	VFMLA V26.S4, V0.S4, V16.S4
	VFMLA V27.S4, V0.S4, V17.S4
	VFMLA V28.S4, V0.S4, V18.S4
	VFMLA V29.S4, V0.S4, V19.S4
	VLD1.P 32(R4), [V30.H8, V31.H8]
	VZIP1 V30.H8, V5.H8, V1.H8
	VZIP2 V30.H8, V5.H8, V2.H8
	VZIP1 V31.H8, V5.H8, V3.H8
	VZIP2 V31.H8, V5.H8, V4.H8
	VFMLA V1.S4, V0.S4, V20.S4
	VFMLA V2.S4, V0.S4, V21.S4
	VFMLA V3.S4, V0.S4, V22.S4
	VFMLA V4.S4, V0.S4, V23.S4
	SUBS $1, R6, R6
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R7)
	VST1.P [V12.S4, V13.S4, V14.S4, V15.S4], 64(R7)
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R7)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R7)
	RET

// func gemv1NEONB(a *float32, b *uint16, panelStride int, k int, out *[16]float32)
//
// gemv1NEON for bf16 panels: per k step, broadcast a[kk] (V0), load one
// 16-wide bf16 row of each panel, widen it with ZIP1/ZIP2 against the zero
// register V5 (as in kernel6x16NEONB) and fused multiply-add it into the
// panel's four accumulators, in the same order as gemv1NEON.
TEXT ·gemv1NEONB(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD panelStride+16(FP), R5
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7
	LSL  $1, R5 // panel stride in bytes

	VEOR V5.B16, V5.B16, V5.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16

loop:
	VLD1R.P 4(R0), [V0.S4]
	VLD1.P 32(R1), [V24.H8, V25.H8]
	VZIP1 V24.H8, V5.H8, V26.H8
	VZIP2 V24.H8, V5.H8, V27.H8
	VZIP1 V25.H8, V5.H8, V28.H8
	VZIP2 V25.H8, V5.H8, V29.H8
	VFMLA V26.S4, V0.S4, V8.S4
	VFMLA V27.S4, V0.S4, V9.S4
	VFMLA V28.S4, V0.S4, V10.S4
	VFMLA V29.S4, V0.S4, V11.S4
	SUBS $1, R6, R6
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R7)
	RET
