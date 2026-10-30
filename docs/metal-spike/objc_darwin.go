// Package main is a throwaway spike: Metal from Go with CGO_ENABLED=0.
// See ../accelerator-plan.md. Run it with `CGO_ENABLED=0 go run .`.
package main

import (
	"syscall"
	"unsafe"
)

// The dynamic imports below are how golang.org/x/sys/unix and crypto/x509
// reach libSystem and Security.framework without cgo: the internal linker
// records the library and symbol, dyld binds them at load time, and an
// assembly trampoline (objc_darwin_arm64.s) jumps to the bound address.

//go:cgo_import_dynamic objc_msgSend objc_msgSend "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_getClass objc_getClass "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic sel_registerName sel_registerName "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_autoreleasePoolPush objc_autoreleasePoolPush "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic objc_autoreleasePoolPop objc_autoreleasePoolPop "/usr/lib/libobjc.A.dylib"
//go:cgo_import_dynamic MTLCreateSystemDefaultDevice MTLCreateSystemDefaultDevice "/System/Library/Frameworks/Metal.framework/Metal"
// loaded.
//go:cgo_import_dynamic _ _ "/System/Library/Frameworks/Foundation.framework/Foundation"

var (
	objc_msgSend_trampoline_addr                 uintptr
	objc_getClass_trampoline_addr                uintptr
	sel_registerName_trampoline_addr             uintptr
	objc_autoreleasePoolPush_trampoline_addr     uintptr
	objc_autoreleasePoolPop_trampoline_addr      uintptr
	MTLCreateSystemDefaultDevice_trampoline_addr uintptr
)

// syscall9 calls a C function on the system stack, telling the scheduler the
// goroutine is blocked: the same entry point x/sys/unix links to.
//
//go:linkname syscall9 syscall.syscall9
//go:noescape
func syscall9(fn, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err syscall.Errno)

type id uintptr

func call(fn uintptr, args ...uintptr) uintptr {
	var a [9]uintptr
	copy(a[:], args)
	r, _, _ := syscall9(fn, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
	return r
}

func cstr(s string) uintptr { b := append([]byte(s), 0); return uintptr(unsafe.Pointer(&b[0])) }

func class(name string) id { return id(call(objc_getClass_trampoline_addr, cstr(name))) }

var sels = map[string]uintptr{}

func sel(name string) uintptr {
	s, ok := sels[name]
	if !ok {
		s = call(sel_registerName_trampoline_addr, cstr(name))
		sels[name] = s
	}
	return s
}

// send is [obj name:args...] for methods whose arguments are all integers or
// pointers (structs over 16 bytes count: arm64 passes them by pointer).
func (o id) send(name string, args ...uintptr) id {
	return id(call(objc_msgSend_trampoline_addr, append([]uintptr{uintptr(o), sel(name)}, args...)...))
}

func nsString(s string) id { return class("NSString").send("stringWithUTF8String:", cstr(s)) }

func goString(o id) string {
	r := o.send("UTF8String")
	p := *(*unsafe.Pointer)(unsafe.Pointer(&r)) // C memory, not Go's
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(p), n))
}
