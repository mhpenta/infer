#include "textflag.h"

TEXT objc_msgSend_trampoline<>(SB),NOSPLIT,$0-0
	JMP	objc_msgSend(SB)
GLOBL	·objc_msgSend_trampoline_addr(SB), RODATA, $8
DATA	·objc_msgSend_trampoline_addr(SB)/8, $objc_msgSend_trampoline<>(SB)

TEXT objc_getClass_trampoline<>(SB),NOSPLIT,$0-0
	JMP	objc_getClass(SB)
GLOBL	·objc_getClass_trampoline_addr(SB), RODATA, $8
DATA	·objc_getClass_trampoline_addr(SB)/8, $objc_getClass_trampoline<>(SB)

TEXT sel_registerName_trampoline<>(SB),NOSPLIT,$0-0
	JMP	sel_registerName(SB)
GLOBL	·sel_registerName_trampoline_addr(SB), RODATA, $8
DATA	·sel_registerName_trampoline_addr(SB)/8, $sel_registerName_trampoline<>(SB)

TEXT objc_autoreleasePoolPush_trampoline<>(SB),NOSPLIT,$0-0
	JMP	objc_autoreleasePoolPush(SB)
GLOBL	·objc_autoreleasePoolPush_trampoline_addr(SB), RODATA, $8
DATA	·objc_autoreleasePoolPush_trampoline_addr(SB)/8, $objc_autoreleasePoolPush_trampoline<>(SB)

TEXT objc_autoreleasePoolPop_trampoline<>(SB),NOSPLIT,$0-0
	JMP	objc_autoreleasePoolPop(SB)
GLOBL	·objc_autoreleasePoolPop_trampoline_addr(SB), RODATA, $8
DATA	·objc_autoreleasePoolPop_trampoline_addr(SB)/8, $objc_autoreleasePoolPop_trampoline<>(SB)

TEXT MTLCreateSystemDefaultDevice_trampoline<>(SB),NOSPLIT,$0-0
	JMP	MTLCreateSystemDefaultDevice(SB)
GLOBL	·MTLCreateSystemDefaultDevice_trampoline_addr(SB), RODATA, $8
DATA	·MTLCreateSystemDefaultDevice_trampoline_addr(SB)/8, $MTLCreateSystemDefaultDevice_trampoline<>(SB)

