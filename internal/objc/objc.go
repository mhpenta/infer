// Package objc calls the Objective-C runtime without cgo, so package metal
// can drive Metal from pure Go.
//
// Every Objective-C method call is a call to one C function,
// objc_msgSend(receiver, selector, args...). On darwin/arm64 this package
// reaches it the way golang.org/x/sys/unix reaches libSystem: a
// //go:cgo_import_dynamic directive has the linker record the symbol, dyld
// binds it at load time, an assembly trampoline jumps to it, and the
// runtime's libc call path (syscall.syscall9) runs it on the system stack.
// On other platforms Available is false and every call returns 0.
//
// Arguments and results are integers or pointers, passed as uintptr: the C
// calling convention puts them in general registers, and arm64 passes
// structs larger than 16 bytes by pointer. Float arguments and results are
// not supported; kernels take their scalars through buffers instead.
//
// Pass a Go pointer as uintptr(unsafe.Pointer(p)) written in the call's
// argument list. Send is marked //go:uintptrescapes, so the compiler moves
// what p points to onto the heap and keeps it alive until the call returns;
// a stack that grows during the call cannot leave C holding a stale address.
// A pointer converted to uintptr earlier, in a variable, has no such
// protection.
package objc

import (
	"runtime"
	"unsafe"
)

// ID is an Objective-C object pointer; 0 is nil.
type ID uintptr

// SEL is a registered method selector.
type SEL uintptr

// cstring returns s as a NUL-terminated byte slice.
func cstring(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// GetClass returns the class named name, or 0 if no loaded image defines it.
func GetClass(name string) ID {
	b := cstring(name)
	return ID(call(getClassAddr, uintptr(unsafe.Pointer(&b[0]))))
}

// RegisterName returns the selector for a method name such as
// "setBuffer:offset:atIndex:". Selectors are interned for the life of the
// process, so callers look them up once.
func RegisterName(name string) SEL {
	b := cstring(name)
	return SEL(call(selRegisterNameAddr, uintptr(unsafe.Pointer(&b[0]))))
}

// Send sends the message sel with args to o and returns the method's integer
// or pointer result (0 for void methods). Sending to nil returns 0.
//
//go:uintptrescapes
func (o ID) Send(sel SEL, args ...uintptr) uintptr {
	if len(args) > 7 {
		panic("objc: Send supports at most 7 arguments")
	}
	var a [9]uintptr
	a[0], a[1] = uintptr(o), uintptr(sel)
	copy(a[2:], args)
	return call(msgSendAddr, a[:]...)
}

// SendBool sends a message whose result is an Objective-C BOOL. Only the low
// byte of the result register is defined.
//
//go:uintptrescapes
func (o ID) SendBool(sel SEL, args ...uintptr) bool {
	return o.Send(sel, args...)&0xff != 0
}

var (
	selRelease              = RegisterName("release")
	selUTF8String           = RegisterName("UTF8String")
	selAlloc                = RegisterName("alloc")
	selInitWithBytes        = RegisterName("initWithBytes:length:encoding:")
	selLocalizedDescription = RegisterName("localizedDescription")
	classNSString           = GetClass("NSString")
)

const nsUTF8StringEncoding = 4

// Release releases an object the caller owns: one returned by a method whose
// name starts with alloc, new, copy or mutableCopy.
func (o ID) Release() {
	if o != 0 {
		o.Send(selRelease)
	}
}

// NewString returns an NSString holding s. The caller owns it.
func NewString(s string) ID {
	str := classNSString.Send(selAlloc)
	return ID(ID(str).Send(selInitWithBytes, uintptr(unsafe.Pointer(unsafe.StringData(s))), uintptr(len(s)), nsUTF8StringEncoding))
}

// String returns the contents of the NSString o, or "" for nil.
func (o ID) String() string {
	return GoString(o.Send(selUTF8String))
}

// Description returns an NSError's localizedDescription, or "" for nil.
func (o ID) Description() string {
	if o == 0 {
		return ""
	}
	return ID(o.Send(selLocalizedDescription)).String()
}

// GoString copies the NUL-terminated C string at p.
func GoString(p uintptr) string {
	if p == 0 {
		return ""
	}
	s := Pointer(p)
	n := 0
	for *(*byte)(unsafe.Add(s, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(s), n))
}

// Pointer converts an address returned by C, which is not Go heap memory,
// to an unsafe.Pointer.
func Pointer(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// Pool is an autorelease pool. Objects returned by methods the caller does
// not own (not alloc, new, copy or mutableCopy) are autoreleased: freed when
// the innermost pool on their thread pops. A pool belongs to an OS thread, so
// PushPool locks the calling goroutine to its thread until Pop.
type Pool uintptr

// PushPool starts an autorelease pool on the calling goroutine's thread.
func PushPool() Pool {
	runtime.LockOSThread()
	return Pool(call(poolPushAddr))
}

// Pop frees the objects autoreleased since p was pushed, and any pools
// pushed after it, then unlocks the thread. It must run on the goroutine that
// pushed p.
func (p Pool) Pop() {
	call(poolPopAddr, uintptr(p))
	runtime.UnlockOSThread()
}
