package objc

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

//go:cgo_import_dynamic objc_msgSend objc_msgSend "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_getClass objc_getClass "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic sel_registerName sel_registerName "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_autoreleasePoolPush objc_autoreleasePoolPush "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_autoreleasePoolPop objc_autoreleasePoolPop "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic MTLCreateSystemDefaultDevice MTLCreateSystemDefaultDevice "/System/Library/Frameworks/Metal.framework/Metal"

// Foundation defines NSString; Metal loads it anyway, but say so.
//go:cgo_import_dynamic _ _ "/System/Library/Frameworks/Foundation.framework/Foundation"

// Available reports whether this build can call the Objective-C runtime.
const Available = true

// The trampolines' addresses, set in objc_darwin_arm64.s.
var (
	objc_msgSend_trampoline_addr                 uintptr
	objc_getClass_trampoline_addr                uintptr
	sel_registerName_trampoline_addr             uintptr
	objc_autoreleasePoolPush_trampoline_addr     uintptr
	objc_autoreleasePoolPop_trampoline_addr      uintptr
	MTLCreateSystemDefaultDevice_trampoline_addr uintptr
)

var (
	msgSendAddr         = objc_msgSend_trampoline_addr
	getClassAddr        = objc_getClass_trampoline_addr
	selRegisterNameAddr = sel_registerName_trampoline_addr
	poolPushAddr        = objc_autoreleasePoolPush_trampoline_addr
	poolPopAddr         = objc_autoreleasePoolPop_trampoline_addr
)

// syscall9 calls the C function fn on the system stack with up to nine
// integer arguments, marking the goroutine as in a syscall so a blocking call
// does not hold up the scheduler. golang.org/x/sys/unix links to it too.
//
//go:linkname syscall9 syscall.syscall9
//go:noescape
func syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err syscall.Errno)

// call calls the C function at fn with up to nine arguments; pointer
// arguments are protected as Send's are.
//
//go:uintptrescapes
func call(fn uintptr, args ...uintptr) uintptr {
	var a [9]uintptr
	copy(a[:], args)
	r, _, _ := syscall9(fn, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
	return r
}

// MTLCreateSystemDefaultDevice returns the default Metal device, owned by
// the caller, or 0 if there is none.
func MTLCreateSystemDefaultDevice() ID {
	return ID(call(MTLCreateSystemDefaultDevice_trampoline_addr))
}
