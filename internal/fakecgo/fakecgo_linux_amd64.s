//go:build !cgo

#include "textflag.h"

// The hooks the runtime calls when iscgo is true. The runtime calls each one
// with the C calling convention on a system stack (from rt0_go, or through
// asmcgocall), so these follow the System V AMD64 ABI: the argument is in DI,
// BX, BP and R12-R15 are callee-saved, and SP is 16-byte aligned at every
// CALL. Calls to glibc go through the PLT entries that the go:cgo_import_dynamic
// directives in fakecgo_linux_amd64.go create.

// The function pointer variables the runtime declares and runtime/cgo would
// otherwise define. They hold static data, so they are set before rt0_go
// reads _cgo_init. The runtime links the environment hooks under its own
// package's names (runtime._cgo_setenv), the others under their C names.
GLOBL	_cgo_init(SB), NOPTR, $8
DATA	_cgo_init(SB)/8, $x_cgo_init<>(SB)
GLOBL	_cgo_thread_start(SB), NOPTR, $8
DATA	_cgo_thread_start(SB)/8, $x_cgo_thread_start<>(SB)
GLOBL	_cgo_notify_runtime_init_done(SB), NOPTR, $8
DATA	_cgo_notify_runtime_init_done(SB)/8, $x_cgo_notify_runtime_init_done<>(SB)
GLOBL	runtime·_cgo_setenv(SB), NOPTR, $8
DATA	runtime·_cgo_setenv(SB)/8, $x_cgo_setenv<>(SB)
GLOBL	runtime·_cgo_unsetenv(SB), NOPTR, $8
DATA	runtime·_cgo_unsetenv(SB)/8, $x_cgo_unsetenv<>(SB)

// _cgo_pthread_key_created points at a flag that stays 0: no pthread key is
// needed because C never calls back into Go.
GLOBL	pthread_key_created<>(SB), NOPTR, $8
GLOBL	_cgo_pthread_key_created(SB), NOPTR, $8
DATA	_cgo_pthread_key_created(SB)/8, $pthread_key_created<>(SB)

// setg_gcc<> holds the runtime's setg_gcc, which stores a g in TLS and R14.
GLOBL	setg_gcc<>(SB), NOPTR, $8

// x_cgo_init(G *g0, void (*setg)(G*)) runs once from rt0_go on the main
// thread. rt0_go has already given g0 the bottom 64 KB of this stack, and
// glibc's loader has already set up TLS, so all that is left is to remember
// setg for new threads.
TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	MOVQ	SI, setg_gcc<>(SB)
	RET

// x_cgo_thread_start(ThreadStart *ts) starts an OS thread for a new M, as
// _cgo_sys_thread_start in runtime/cgo does. ts is {g *G, tls *uintptr,
// fn uintptr} and lives on the caller's stack, so it is copied to the C heap
// for the new thread, which frees it. Signals stay blocked while the thread
// is created, so it starts with all of them blocked until minit sets its
// mask. Failure to start a thread aborts, as it does under cgo.
//
// Frame, from SP: new sigset_t (128 bytes) at 0, old sigset_t at 128,
// pthread_attr_t (56 bytes) at 256, pthread_t at 320, stack size at 328.
TEXT x_cgo_thread_start<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	PUSHQ	R12
	PUSHQ	R13
	SUBQ	$344, SP
	MOVQ	DI, BX

	MOVQ	$24, DI
	CALL	malloc(SB)
	TESTQ	AX, AX
	JZ	fail
	MOVQ	0(BX), CX
	MOVQ	CX, 0(AX)
	MOVQ	8(BX), CX
	MOVQ	CX, 8(AX)
	MOVQ	16(BX), CX
	MOVQ	CX, 16(AX)
	MOVQ	AX, R12

	LEAQ	0(SP), DI
	CALL	sigfillset(SB)
	MOVL	$2, DI // SIG_SETMASK
	LEAQ	0(SP), SI
	LEAQ	128(SP), DX
	CALL	pthread_sigmask(SB)

	LEAQ	256(SP), DI
	CALL	pthread_attr_init(SB)
	LEAQ	256(SP), DI
	LEAQ	328(SP), SI
	CALL	pthread_attr_getstacksize(SB)
	// Leave g0.stack.lo 0 and put the stack size in g0.stack.hi (offset 8
	// in g); mstart0 turns that into the stack's bounds.
	MOVQ	0(R12), AX
	MOVQ	328(SP), CX
	MOVQ	CX, 8(AX)

	LEAQ	320(SP), DI
	LEAQ	256(SP), SI
	MOVQ	$threadentry<>(SB), DX
	MOVQ	R12, CX
	CALL	pthread_create(SB)
	MOVL	AX, R13

	MOVL	$2, DI // SIG_SETMASK
	LEAQ	128(SP), SI
	XORL	DX, DX
	CALL	pthread_sigmask(SB)
	LEAQ	256(SP), DI
	CALL	pthread_attr_destroy(SB)

	TESTL	R13, R13
	JNZ	fail
	MOVQ	320(SP), DI
	CALL	pthread_detach(SB)

	ADDQ	$344, SP
	POPQ	R13
	POPQ	R12
	POPQ	BX
	POPQ	BP
	RET

fail:
	CALL	abort(SB)
	RET

// threadentry(ThreadStart *ts) is the new thread's start routine. It frees
// the copy of ts, makes ts.g the current g, and runs ts.fn (mstart), which
// returns when the M exits; returning from here then ends the thread.
TEXT threadentry<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	PUSHQ	BX
	PUSHQ	R12
	PUSHQ	R13
	PUSHQ	R14
	PUSHQ	R15
	SUBQ	$8, SP

	MOVQ	0(DI), R12  // g
	MOVQ	16(DI), R13 // fn
	CALL	free(SB)
	MOVQ	R12, DI
	MOVQ	setg_gcc<>(SB), AX
	CALL	AX
	CALL	R13

	ADDQ	$8, SP
	POPQ	R15
	POPQ	R14
	POPQ	R13
	POPQ	R12
	POPQ	BX
	POPQ	BP
	XORL	AX, AX
	RET

// x_cgo_notify_runtime_init_done would wake C threads waiting to call into
// Go. There are none.
TEXT x_cgo_notify_runtime_init_done<>(SB),NOSPLIT|NOFRAME,$0-0
	RET

// x_cgo_setenv(char **kv) mirrors os.Setenv into the C environment, so C
// libraries see it: setenv(kv[0], kv[1], 1).
TEXT x_cgo_setenv<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	8(DI), SI
	MOVQ	0(DI), DI
	MOVL	$1, DX
	CALL	setenv(SB)
	POPQ	BP
	RET

// x_cgo_unsetenv(char **k) mirrors os.Unsetenv: unsetenv(k[0]).
TEXT x_cgo_unsetenv<>(SB),NOSPLIT|NOFRAME,$0-0
	PUSHQ	BP
	MOVQ	SP, BP
	MOVQ	0(DI), DI
	CALL	unsetenv(SB)
	POPQ	BP
	RET
