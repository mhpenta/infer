//go:build arm64 && !purego

#include "textflag.h"

// func gemv4NEON(a, b *float32, panelStride int, k int, out *[64]float32)
//
// out[16p + j] = sum over kk < k of a[kk]·b[p][kk][j] for 4 panel(s), k >= 1:
// per k step, broadcast a[kk] (V0) and fused multiply-add one 16-wide row of
// each panel into its four accumulators. Each output is one sequential FMA
// chain, as in kernel6x16NEON, so results are bit-identical to it.
TEXT ·gemv4NEON(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD panelStride+16(FP), R5
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7
	LSL  $2, R5
	ADD  R5, R1, R2
	ADD  R5, R2, R3
	ADD  R5, R3, R4

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
	VLD1.P 64(R1), [V1.S4, V2.S4, V3.S4, V4.S4]
	VFMLA V1.S4, V0.S4, V8.S4
	VFMLA V2.S4, V0.S4, V9.S4
	VFMLA V3.S4, V0.S4, V10.S4
	VFMLA V4.S4, V0.S4, V11.S4
	VLD1.P 64(R2), [V24.S4, V25.S4, V26.S4, V27.S4]
	VFMLA V24.S4, V0.S4, V12.S4
	VFMLA V25.S4, V0.S4, V13.S4
	VFMLA V26.S4, V0.S4, V14.S4
	VFMLA V27.S4, V0.S4, V15.S4
	VLD1.P 64(R3), [V28.S4, V29.S4, V30.S4, V31.S4]
	VFMLA V28.S4, V0.S4, V16.S4
	VFMLA V29.S4, V0.S4, V17.S4
	VFMLA V30.S4, V0.S4, V18.S4
	VFMLA V31.S4, V0.S4, V19.S4
	VLD1.P 64(R4), [V1.S4, V2.S4, V3.S4, V4.S4]
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

// func gemv1NEON(a, b *float32, panelStride int, k int, out *[16]float32)
//
// out[16p + j] = sum over kk < k of a[kk]·b[p][kk][j] for 1 panel(s), k >= 1:
// per k step, broadcast a[kk] (V0) and fused multiply-add one 16-wide row of
// each panel into its four accumulators. Each output is one sequential FMA
// chain, as in kernel6x16NEON, so results are bit-identical to it.
TEXT ·gemv1NEON(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD panelStride+16(FP), R5
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7


	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16

loop:
	VLD1R.P 4(R0), [V0.S4]
	VLD1.P 64(R1), [V1.S4, V2.S4, V3.S4, V4.S4]
	VFMLA V1.S4, V0.S4, V8.S4
	VFMLA V2.S4, V0.S4, V9.S4
	VFMLA V3.S4, V0.S4, V10.S4
	VFMLA V4.S4, V0.S4, V11.S4
	SUBS $1, R6, R6
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R7)
	RET
