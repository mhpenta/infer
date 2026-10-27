//go:build arm64 && !purego

#include "textflag.h"

// NEON kernels for int8 panels: kernel_arm64.s and gemv_arm64.s with each
// panel row widened from int8 to float32 in registers. Go's assembler lacks
// the signed widening shift, the vector integer-to-float conversion and the
// vector float subtract, so the macros below encode them by hand (checked
// against clang's encodings). They take vector register numbers.

// SXTL_8H(d, n): sign-extend the low 8 bytes of Vn into the 8 halfwords of
// Vd (SSHLL Vd.8H, Vn.8B, #0); SXTL2_8H the high 8 bytes (SSHLL2).
#define SXTL_8H(d, n) WORD $(0x0F08A400 | (n)<<5 | (d))
#define SXTL2_8H(d, n) WORD $(0x4F08A400 | (n)<<5 | (d))

// SXTL_4S(d, n): sign-extend the low 4 halfwords of Vn into the 4 words of
// Vd (SSHLL Vd.4S, Vn.4H, #0); SXTL2_4S the high 4 (SSHLL2).
#define SXTL_4S(d, n) WORD $(0x0F10A400 | (n)<<5 | (d))
#define SXTL2_4S(d, n) WORD $(0x4F10A400 | (n)<<5 | (d))

// SCVTF_4S(d, n): convert the 4 signed words of Vn to float32 in Vd
// (SCVTF Vd.4S, Vn.4S), exact for int8 values.
#define SCVTF_4S(d, n) WORD $(0x4E21D800 | (n)<<5 | (d))

// FSUB_4S(d, n, m): Vd.4S = Vn.4S - Vm.4S.
#define FSUB_4S(d, n, m) WORD $(0x4EA0D400 | (m)<<16 | (n)<<5 | (d))

// int8Spread holds the four TBL index vectors of the one-row kernels. Vector
// v places bytes 4v to 4v+3 of a panel row in the low byte of four 32-bit
// lanes, with table bytes 16, 16 and 17 above (0x00, 0x00 and 0x4B; see
// gemv4NEONI).
DATA int8Spread<>+0(SB)/8, $0x1110100111101000
DATA int8Spread<>+8(SB)/8, $0x1110100311101002
DATA int8Spread<>+16(SB)/8, $0x1110100511101004
DATA int8Spread<>+24(SB)/8, $0x1110100711101006
DATA int8Spread<>+32(SB)/8, $0x1110100911101008
DATA int8Spread<>+40(SB)/8, $0x1110100b1110100a
DATA int8Spread<>+48(SB)/8, $0x1110100d1110100c
DATA int8Spread<>+56(SB)/8, $0x1110100f1110100e
GLOBL int8Spread<>(SB), RODATA|NOPTR, $64

// func kernel6x16NEONI(a *float32, b *int8, k int, out *[96]float32)
//
// kernel6x16NEON for an int8 panel. Per k step it loads one 16-wide panel
// row of int8 values (V4), sign-extends it to halfwords (V0, V1) and then to
// words in V2, V3, V0, V1 (columns 0-3, 4-7, 8-11, 12-15), and converts
// those to float32. It then broadcasts the 6 values of the packed a strip
// (V5-V7, rotating) and issues 24 fused multiply-adds into V8-V31 in the
// same order as kernel6x16NEON, so results are bit-identical to its on the
// converted values.
TEXT ·kernel6x16NEONI(SB), NOSPLIT, $0-32
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
	VLD1.P 16(R7), [V4.B16]
	SXTL_8H(0, 4)
	SXTL2_8H(1, 4)
	SXTL_4S(2, 0)
	SXTL2_4S(3, 0)
	SXTL_4S(0, 1)
	SXTL2_4S(1, 1)
	SCVTF_4S(2, 2)
	SCVTF_4S(3, 3)
	SCVTF_4S(0, 0)
	SCVTF_4S(1, 1)

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

// func gemv4NEONI(a *float32, b *int8, panelStride int, k int, out *[64]float32)
//
// gemv4NEON for int8 panels. Widening each value with conversions, as
// kernel6x16NEONI does, would cost 10 instructions per panel row and limit a
// single row's product, which uses every value once; this builds the floats
// from bytes instead. Per k step: broadcast a[kk] (V0); then per panel, load
// one 16-wide int8 row (V24) and flip each byte's sign bit (V6), giving
// u = q + 128 in [1, 255]. Four TBLs over the table {V24, V25} (V1-V4, see
// int8Spread) write lanes with bytes u, 0x00, 0x00, 0x4B: the float32
// 2^23 + u. Subtracting 2^23 + 128 (V5) leaves q exactly, as a float32, in
// V26-V29 (columns 0-3, 4-7, 8-11, 12-15), which are fused multiply-added
// into the panel's four accumulators in the same order as gemv4NEON.
TEXT ·gemv4NEONI(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD panelStride+16(FP), R5 // panel stride: values are bytes
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7
	ADD  R5, R1, R2
	ADD  R5, R2, R3
	ADD  R5, R3, R4
	MOVD $int8Spread<>(SB), R8
	VLD1 (R8), [V1.B16, V2.B16, V3.B16, V4.B16]
	MOVW $0x4B000080, R8 // 2^23 + 128 as float32
	VDUP R8, V5.S4
	MOVW $0x80808080, R8
	VDUP R8, V6.S4
	MOVW $0x4B00, R8 // table bytes 16 and 17: 0x00, 0x4B
	VDUP R8, V25.S4

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
	VLD1.P 16(R1), [V24.B16]
	VEOR   V6.B16, V24.B16, V24.B16
	VTBL   V1.B16, [V24.B16, V25.B16], V26.B16
	VTBL   V2.B16, [V24.B16, V25.B16], V27.B16
	VTBL   V3.B16, [V24.B16, V25.B16], V28.B16
	VTBL   V4.B16, [V24.B16, V25.B16], V29.B16
	FSUB_4S(26, 26, 5)
	FSUB_4S(27, 27, 5)
	FSUB_4S(28, 28, 5)
	FSUB_4S(29, 29, 5)
	VFMLA  V26.S4, V0.S4, V8.S4
	VFMLA  V27.S4, V0.S4, V9.S4
	VFMLA  V28.S4, V0.S4, V10.S4
	VFMLA  V29.S4, V0.S4, V11.S4
	VLD1.P 16(R2), [V24.B16]
	VEOR   V6.B16, V24.B16, V24.B16
	VTBL   V1.B16, [V24.B16, V25.B16], V26.B16
	VTBL   V2.B16, [V24.B16, V25.B16], V27.B16
	VTBL   V3.B16, [V24.B16, V25.B16], V28.B16
	VTBL   V4.B16, [V24.B16, V25.B16], V29.B16
	FSUB_4S(26, 26, 5)
	FSUB_4S(27, 27, 5)
	FSUB_4S(28, 28, 5)
	FSUB_4S(29, 29, 5)
	VFMLA  V26.S4, V0.S4, V12.S4
	VFMLA  V27.S4, V0.S4, V13.S4
	VFMLA  V28.S4, V0.S4, V14.S4
	VFMLA  V29.S4, V0.S4, V15.S4
	VLD1.P 16(R3), [V24.B16]
	VEOR   V6.B16, V24.B16, V24.B16
	VTBL   V1.B16, [V24.B16, V25.B16], V26.B16
	VTBL   V2.B16, [V24.B16, V25.B16], V27.B16
	VTBL   V3.B16, [V24.B16, V25.B16], V28.B16
	VTBL   V4.B16, [V24.B16, V25.B16], V29.B16
	FSUB_4S(26, 26, 5)
	FSUB_4S(27, 27, 5)
	FSUB_4S(28, 28, 5)
	FSUB_4S(29, 29, 5)
	VFMLA  V26.S4, V0.S4, V16.S4
	VFMLA  V27.S4, V0.S4, V17.S4
	VFMLA  V28.S4, V0.S4, V18.S4
	VFMLA  V29.S4, V0.S4, V19.S4
	VLD1.P 16(R4), [V24.B16]
	VEOR   V6.B16, V24.B16, V24.B16
	VTBL   V1.B16, [V24.B16, V25.B16], V26.B16
	VTBL   V2.B16, [V24.B16, V25.B16], V27.B16
	VTBL   V3.B16, [V24.B16, V25.B16], V28.B16
	VTBL   V4.B16, [V24.B16, V25.B16], V29.B16
	FSUB_4S(26, 26, 5)
	FSUB_4S(27, 27, 5)
	FSUB_4S(28, 28, 5)
	FSUB_4S(29, 29, 5)
	VFMLA  V26.S4, V0.S4, V20.S4
	VFMLA  V27.S4, V0.S4, V21.S4
	VFMLA  V28.S4, V0.S4, V22.S4
	VFMLA  V29.S4, V0.S4, V23.S4
	SUBS $1, R6, R6
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R7)
	VST1.P [V12.S4, V13.S4, V14.S4, V15.S4], 64(R7)
	VST1.P [V16.S4, V17.S4, V18.S4, V19.S4], 64(R7)
	VST1.P [V20.S4, V21.S4, V22.S4, V23.S4], 64(R7)
	RET

// func gemv1NEONI(a *float32, b *int8, panelStride int, k int, out *[16]float32)
//
// gemv1NEON for int8 panels: gemv4NEONI's step for a single panel.
TEXT ·gemv1NEONI(SB), NOSPLIT, $0-40
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD k+24(FP), R6
	MOVD out+32(FP), R7
	MOVD $int8Spread<>(SB), R8
	VLD1 (R8), [V1.B16, V2.B16, V3.B16, V4.B16]
	MOVW $0x4B000080, R8 // 2^23 + 128 as float32
	VDUP R8, V5.S4
	MOVW $0x80808080, R8
	VDUP R8, V6.S4
	MOVW $0x4B00, R8 // table bytes 16 and 17: 0x00, 0x4B
	VDUP R8, V25.S4

	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16

loop:
	VLD1R.P 4(R0), [V0.S4]
	VLD1.P 16(R1), [V24.B16]
	VEOR   V6.B16, V24.B16, V24.B16
	VTBL   V1.B16, [V24.B16, V25.B16], V26.B16
	VTBL   V2.B16, [V24.B16, V25.B16], V27.B16
	VTBL   V3.B16, [V24.B16, V25.B16], V28.B16
	VTBL   V4.B16, [V24.B16, V25.B16], V29.B16
	FSUB_4S(26, 26, 5)
	FSUB_4S(27, 27, 5)
	FSUB_4S(28, 28, 5)
	FSUB_4S(29, 29, 5)
	VFMLA  V26.S4, V0.S4, V8.S4
	VFMLA  V27.S4, V0.S4, V9.S4
	VFMLA  V28.S4, V0.S4, V10.S4
	VFMLA  V29.S4, V0.S4, V11.S4
	SUBS $1, R6, R6
	BNE  loop

	VST1.P [V8.S4, V9.S4, V10.S4, V11.S4], 64(R7)
	RET
