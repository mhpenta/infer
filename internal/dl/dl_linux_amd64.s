//go:build !cgo

#include "textflag.h"

// Each trampoline jumps to a symbol bound by the dynamic loader (see the
// go:cgo_import_dynamic directives in dl_linux_amd64.go); its address is
// stored in a Go variable for Call.

TEXT dlopen_trampoline<>(SB),NOSPLIT|NOFRAME,$0-0
	JMP	dlopen(SB)
GLOBL	·dlopenAddr(SB), RODATA, $8
DATA	·dlopenAddr(SB)/8, $dlopen_trampoline<>(SB)

TEXT dlsym_trampoline<>(SB),NOSPLIT|NOFRAME,$0-0
	JMP	dlsym(SB)
GLOBL	·dlsymAddr(SB), RODATA, $8
DATA	·dlsymAddr(SB)/8, $dlsym_trampoline<>(SB)

TEXT dlerror_trampoline<>(SB),NOSPLIT|NOFRAME,$0-0
	JMP	dlerror(SB)
GLOBL	·dlerrorAddr(SB), RODATA, $8
DATA	·dlerrorAddr(SB)/8, $dlerror_trampoline<>(SB)

// callTrampoline(frame *f) calls f.fn(f.args...) with the System V AMD64
// calling convention and stores the result in f.r1. asmcgocall runs it on
// the system stack with f in DI and SP 16-byte aligned before its CALL.
// Arguments 1-6 go in registers and 7-12 on the stack. AL is zeroed, the
// count of vector registers a variadic callee reads.
//
// frame layout: fn at 0, args[i] at 8+8*i, r1 at 104.
TEXT callTrampoline<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	SUBQ	$56, SP
	MOVQ	DI, BX

	MOVQ	56(BX), AX
	MOVQ	AX, 0(SP)
	MOVQ	64(BX), AX
	MOVQ	AX, 8(SP)
	MOVQ	72(BX), AX
	MOVQ	AX, 16(SP)
	MOVQ	80(BX), AX
	MOVQ	AX, 24(SP)
	MOVQ	88(BX), AX
	MOVQ	AX, 32(SP)
	MOVQ	96(BX), AX
	MOVQ	AX, 40(SP)

	MOVQ	8(BX), DI
	MOVQ	16(BX), SI
	MOVQ	24(BX), DX
	MOVQ	32(BX), CX
	MOVQ	40(BX), R8
	MOVQ	48(BX), R9
	MOVQ	0(BX), R10
	XORL	AX, AX
	CALL	R10
	MOVQ	AX, 104(BX)

	ADDQ	$56, SP
	POPQ	BX
	POPQ	BP
	RET
GLOBL	·callTrampolineAddr(SB), RODATA, $8
DATA	·callTrampolineAddr(SB)/8, $callTrampoline<>(SB)
