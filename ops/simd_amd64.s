//go:build amd64 && !purego

#include "textflag.h"

// Vector loops behind softmax and GELU, 8 float32 lanes at a time. Every
// function takes n, a positive multiple of 8; the Go wrappers handle the tail.
// Constants are stored replicated across a full YMM register so instructions
// can read them straight from memory.
//
// Generated layout; each constant is 32 bytes.
// log2e = 1.4426950408889634
DATA vconst<>+0(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA vconst<>+8(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA vconst<>+16(SB)/8, $0x3fb8aa3b3fb8aa3b
DATA vconst<>+24(SB)/8, $0x3fb8aa3b3fb8aa3b
// half = 0.5
DATA vconst<>+32(SB)/8, $0x3f0000003f000000
DATA vconst<>+40(SB)/8, $0x3f0000003f000000
DATA vconst<>+48(SB)/8, $0x3f0000003f000000
DATA vconst<>+56(SB)/8, $0x3f0000003f000000
// one = 1.0
DATA vconst<>+64(SB)/8, $0x3f8000003f800000
DATA vconst<>+72(SB)/8, $0x3f8000003f800000
DATA vconst<>+80(SB)/8, $0x3f8000003f800000
DATA vconst<>+88(SB)/8, $0x3f8000003f800000
// ln2Hi = 0.693359375
DATA vconst<>+96(SB)/8, $0x3f3180003f318000
DATA vconst<>+104(SB)/8, $0x3f3180003f318000
DATA vconst<>+112(SB)/8, $0x3f3180003f318000
DATA vconst<>+120(SB)/8, $0x3f3180003f318000
// ln2Lo = -0.00021219444
DATA vconst<>+128(SB)/8, $0xb95e8083b95e8083
DATA vconst<>+136(SB)/8, $0xb95e8083b95e8083
DATA vconst<>+144(SB)/8, $0xb95e8083b95e8083
DATA vconst<>+152(SB)/8, $0xb95e8083b95e8083
// maxArg = 88.3762626647949
DATA vconst<>+160(SB)/8, $0x42b0c0a542b0c0a5
DATA vconst<>+168(SB)/8, $0x42b0c0a542b0c0a5
DATA vconst<>+176(SB)/8, $0x42b0c0a542b0c0a5
DATA vconst<>+184(SB)/8, $0x42b0c0a542b0c0a5
// minArg = -87.3365447505531
DATA vconst<>+192(SB)/8, $0xc2aeac50c2aeac50
DATA vconst<>+200(SB)/8, $0xc2aeac50c2aeac50
DATA vconst<>+208(SB)/8, $0xc2aeac50c2aeac50
DATA vconst<>+216(SB)/8, $0xc2aeac50c2aeac50
// p0 = 0.000198756915
DATA vconst<>+224(SB)/8, $0x3950696739506967
DATA vconst<>+232(SB)/8, $0x3950696739506967
DATA vconst<>+240(SB)/8, $0x3950696739506967
DATA vconst<>+248(SB)/8, $0x3950696739506967
// p1 = 0.0013981999507
DATA vconst<>+256(SB)/8, $0x3ab743ce3ab743ce
DATA vconst<>+264(SB)/8, $0x3ab743ce3ab743ce
DATA vconst<>+272(SB)/8, $0x3ab743ce3ab743ce
DATA vconst<>+280(SB)/8, $0x3ab743ce3ab743ce
// p2 = 0.0083334519073
DATA vconst<>+288(SB)/8, $0x3c0889083c088908
DATA vconst<>+296(SB)/8, $0x3c0889083c088908
DATA vconst<>+304(SB)/8, $0x3c0889083c088908
DATA vconst<>+312(SB)/8, $0x3c0889083c088908
// p3 = 0.041665795894
DATA vconst<>+320(SB)/8, $0x3d2aa9c13d2aa9c1
DATA vconst<>+328(SB)/8, $0x3d2aa9c13d2aa9c1
DATA vconst<>+336(SB)/8, $0x3d2aa9c13d2aa9c1
DATA vconst<>+344(SB)/8, $0x3d2aa9c13d2aa9c1
// p4 = 0.16666665459
DATA vconst<>+352(SB)/8, $0x3e2aaaaa3e2aaaaa
DATA vconst<>+360(SB)/8, $0x3e2aaaaa3e2aaaaa
DATA vconst<>+368(SB)/8, $0x3e2aaaaa3e2aaaaa
DATA vconst<>+376(SB)/8, $0x3e2aaaaa3e2aaaaa
// p5 = 0.50000001201
DATA vconst<>+384(SB)/8, $0x3f0000003f000000
DATA vconst<>+392(SB)/8, $0x3f0000003f000000
DATA vconst<>+400(SB)/8, $0x3f0000003f000000
DATA vconst<>+408(SB)/8, $0x3f0000003f000000
// four = 4.0
DATA vconst<>+416(SB)/8, $0x4080000040800000
DATA vconst<>+424(SB)/8, $0x4080000040800000
DATA vconst<>+432(SB)/8, $0x4080000040800000
DATA vconst<>+440(SB)/8, $0x4080000040800000
// mfour = -4.0
DATA vconst<>+448(SB)/8, $0xc0800000c0800000
DATA vconst<>+456(SB)/8, $0xc0800000c0800000
DATA vconst<>+464(SB)/8, $0xc0800000c0800000
DATA vconst<>+472(SB)/8, $0xc0800000c0800000
// rsqrt2 = 0.7071067811865476
DATA vconst<>+480(SB)/8, $0x3f3504f33f3504f3
DATA vconst<>+488(SB)/8, $0x3f3504f33f3504f3
DATA vconst<>+496(SB)/8, $0x3f3504f33f3504f3
DATA vconst<>+504(SB)/8, $0x3f3504f33f3504f3
// a13 = -2.72614225801306e-10
DATA vconst<>+512(SB)/8, $0xaf95df0baf95df0b
DATA vconst<>+520(SB)/8, $0xaf95df0baf95df0b
DATA vconst<>+528(SB)/8, $0xaf95df0baf95df0b
DATA vconst<>+536(SB)/8, $0xaf95df0baf95df0b
// a11 = 2.77068142495902e-08
DATA vconst<>+544(SB)/8, $0x32edffee32edffee
DATA vconst<>+552(SB)/8, $0x32edffee32edffee
DATA vconst<>+560(SB)/8, $0x32edffee32edffee
DATA vconst<>+568(SB)/8, $0x32edffee32edffee
// a9 = -2.10102402082508e-06
DATA vconst<>+576(SB)/8, $0xb60cff51b60cff51
DATA vconst<>+584(SB)/8, $0xb60cff51b60cff51
DATA vconst<>+592(SB)/8, $0xb60cff51b60cff51
DATA vconst<>+600(SB)/8, $0xb60cff51b60cff51
// a7 = -5.69250639462346e-05
DATA vconst<>+608(SB)/8, $0xb86ec2d2b86ec2d2
DATA vconst<>+616(SB)/8, $0xb86ec2d2b86ec2d2
DATA vconst<>+624(SB)/8, $0xb86ec2d2b86ec2d2
DATA vconst<>+632(SB)/8, $0xb86ec2d2b86ec2d2
// a5 = -0.000734990630326855
DATA vconst<>+640(SB)/8, $0xba40ac63ba40ac63
DATA vconst<>+648(SB)/8, $0xba40ac63ba40ac63
DATA vconst<>+656(SB)/8, $0xba40ac63ba40ac63
DATA vconst<>+664(SB)/8, $0xba40ac63ba40ac63
// a3 = -0.00295459980854025
DATA vconst<>+672(SB)/8, $0xbb41a1f6bb41a1f6
DATA vconst<>+680(SB)/8, $0xbb41a1f6bb41a1f6
DATA vconst<>+688(SB)/8, $0xbb41a1f6bb41a1f6
DATA vconst<>+696(SB)/8, $0xbb41a1f6bb41a1f6
// a1 = -0.0160960333262415
DATA vconst<>+704(SB)/8, $0xbc83dbd4bc83dbd4
DATA vconst<>+712(SB)/8, $0xbc83dbd4bc83dbd4
DATA vconst<>+720(SB)/8, $0xbc83dbd4bc83dbd4
DATA vconst<>+728(SB)/8, $0xbc83dbd4bc83dbd4
// b8 = -1.45660718464996e-05
DATA vconst<>+736(SB)/8, $0xb77460cdb77460cd
DATA vconst<>+744(SB)/8, $0xb77460cdb77460cd
DATA vconst<>+752(SB)/8, $0xb77460cdb77460cd
DATA vconst<>+760(SB)/8, $0xb77460cdb77460cd
// b6 = -0.000213374055278905
DATA vconst<>+768(SB)/8, $0xb95fbd29b95fbd29
DATA vconst<>+776(SB)/8, $0xb95fbd29b95fbd29
DATA vconst<>+784(SB)/8, $0xb95fbd29b95fbd29
DATA vconst<>+792(SB)/8, $0xb95fbd29b95fbd29
// b4 = -0.00168282697438203
DATA vconst<>+800(SB)/8, $0xbadc924ebadc924e
DATA vconst<>+808(SB)/8, $0xbadc924ebadc924e
DATA vconst<>+816(SB)/8, $0xbadc924ebadc924e
DATA vconst<>+824(SB)/8, $0xbadc924ebadc924e
// b2 = -0.00737332916720468
DATA vconst<>+832(SB)/8, $0xbbf19bf8bbf19bf8
DATA vconst<>+840(SB)/8, $0xbbf19bf8bbf19bf8
DATA vconst<>+848(SB)/8, $0xbbf19bf8bbf19bf8
DATA vconst<>+856(SB)/8, $0xbbf19bf8bbf19bf8
// b0 = -0.0142647390514189
DATA vconst<>+864(SB)/8, $0xbc69b6a7bc69b6a7
DATA vconst<>+872(SB)/8, $0xbc69b6a7bc69b6a7
DATA vconst<>+880(SB)/8, $0xbc69b6a7bc69b6a7
DATA vconst<>+888(SB)/8, $0xbc69b6a7bc69b6a7
// bias127 = 127 (int32), the float32 exponent bias
DATA vconst<>+896(SB)/8, $0x0000007f0000007f
DATA vconst<>+904(SB)/8, $0x0000007f0000007f
DATA vconst<>+912(SB)/8, $0x0000007f0000007f
DATA vconst<>+920(SB)/8, $0x0000007f0000007f
GLOBL vconst<>(SB), RODATA|NOPTR, $928

// func maxAVX2(x *float32, n int, out *[8]float32)
TEXT ·maxAVX2(SB), NOSPLIT, $0-24
	MOVQ    x+0(FP), SI
	MOVQ    n+8(FP), CX
	MOVQ    out+16(FP), DI
	VMOVUPS (SI), Y0
	MOVQ    $8, AX

maxloop:
	CMPQ   AX, CX
	JGE    maxdone
	VMAXPS (SI)(AX*4), Y0, Y0
	ADDQ   $8, AX
	JMP    maxloop

maxdone:
	VMOVUPS Y0, (DI)
	VZEROUPPER
	RET

// func expShiftScaleAVX2(x *float32, n int, shift, scale float32, sums *[8]float32)
//
// x[i] = e**((x[i]-shift)·scale), with lane sums of the results in sums. The
// exponential is expf's: n = floor(t·log2e + 0.5), r = t - n·ln2 (split in
// two parts), a degree-5 polynomial for e**r, then 2**n through the exponent
// bits. Inputs below minArg give exactly 0; inputs are clamped at maxArg.
TEXT ·expShiftScaleAVX2(SB), NOSPLIT, $0-32
	MOVQ         x+0(FP), SI
	MOVQ         n+8(FP), CX
	VBROADCASTSS shift+16(FP), Y14
	VBROADCASTSS scale+20(FP), Y15
	MOVQ         sums+24(FP), DI
	VXORPS       Y13, Y13, Y13
	XORQ         AX, AX

exploop:
	VMOVUPS (SI)(AX*4), Y0
	VSUBPS  Y14, Y0, Y0
	VMULPS  Y15, Y0, Y0

	VMINPS vconst<>+160(SB), Y0, Y0
	VCMPPS $1, vconst<>+192(SB), Y0, Y5 // Y5 = lanes below minArg

	VMOVUPS      vconst<>+32(SB), Y1
	VFMADD231PS  vconst<>+0(SB), Y0, Y1 // t = x·log2e + 0.5
	VROUNDPS     $1, Y1, Y1           // n = floor(t)
	VFNMADD231PS vconst<>+96(SB), Y1, Y0 // r = x - n·ln2Hi
	VFNMADD231PS vconst<>+128(SB), Y1, Y0 // r -= n·ln2Lo

	VMOVUPS     vconst<>+224(SB), Y2
	VFMADD213PS vconst<>+256(SB), Y0, Y2
	VFMADD213PS vconst<>+288(SB), Y0, Y2
	VFMADD213PS vconst<>+320(SB), Y0, Y2
	VFMADD213PS vconst<>+352(SB), Y0, Y2
	VFMADD213PS vconst<>+384(SB), Y0, Y2

	VMULPS      Y0, Y0, Y3       // r²
	VADDPS      vconst<>+64(SB), Y0, Y4 // r + 1
	VFMADD231PS Y3, Y2, Y4       // y = p·r² + r + 1

	VCVTPS2DQ Y1, Y1
	VPADDD    vconst<>+896(SB), Y1, Y1
	VPSLLD    $23, Y1, Y1 // 2**n
	VMULPS    Y1, Y4, Y0
	VANDNPS   Y0, Y5, Y0  // 0 below minArg

	VMOVUPS Y0, (SI)(AX*4)
	VADDPS  Y0, Y13, Y13
	ADDQ    $8, AX
	CMPQ    AX, CX
	JLT     exploop

	VMOVUPS Y13, (DI)
	VZEROUPPER
	RET

// func scaleAVX2(x *float32, n int, s float32)
TEXT ·scaleAVX2(SB), NOSPLIT, $0-20
	MOVQ         x+0(FP), SI
	MOVQ         n+8(FP), CX
	VBROADCASTSS s+16(FP), Y1
	XORQ         AX, AX

scaleloop:
	VMULPS  (SI)(AX*4), Y1, Y0
	VMOVUPS Y0, (SI)(AX*4)
	ADDQ    $8, AX
	CMPQ    AX, CX
	JLT     scaleloop

	VZEROUPPER
	RET

// func geluAVX2(x *float32, n int)
//
// x = 0.5·x·(1 + erf(x/√2)), with erff's rational approximation: u clamped
// to [-4, 4], erf(u) = u·P(u²) / Q(u²).
TEXT ·geluAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	XORQ AX, AX

geluloop:
	VMOVUPS (SI)(AX*4), Y0
	VMULPS  vconst<>+480(SB), Y0, Y1
	VMINPS  vconst<>+416(SB), Y1, Y1
	VMAXPS  vconst<>+448(SB), Y1, Y1
	VMULPS  Y1, Y1, Y2 // u²

	VMOVUPS     vconst<>+512(SB), Y3
	VFMADD213PS vconst<>+544(SB), Y2, Y3
	VFMADD213PS vconst<>+576(SB), Y2, Y3
	VFMADD213PS vconst<>+608(SB), Y2, Y3
	VFMADD213PS vconst<>+640(SB), Y2, Y3
	VFMADD213PS vconst<>+672(SB), Y2, Y3
	VFMADD213PS vconst<>+704(SB), Y2, Y3
	VMULPS      Y1, Y3, Y3 // P

	VMOVUPS     vconst<>+736(SB), Y4
	VFMADD213PS vconst<>+768(SB), Y2, Y4
	VFMADD213PS vconst<>+800(SB), Y2, Y4
	VFMADD213PS vconst<>+832(SB), Y2, Y4
	VFMADD213PS vconst<>+864(SB), Y2, Y4 // Q

	VDIVPS Y4, Y3, Y3 // erf
	VADDPS vconst<>+64(SB), Y3, Y3
	VMULPS vconst<>+32(SB), Y0, Y0
	VMULPS Y3, Y0, Y0

	VMOVUPS Y0, (SI)(AX*4)
	ADDQ    $8, AX
	CMPQ    AX, CX
	JLT     geluloop

	VZEROUPPER
	RET

