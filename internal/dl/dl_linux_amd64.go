//go:build !cgo

package dl

import (
	"fmt"
	"runtime"
	"unsafe"

	_ "github.com/mhpenta/infer/internal/fakecgo" // start threads with pthread_create
)

//go:cgo_import_dynamic dlopen dlopen "libdl.so.2"
//go:cgo_import_dynamic dlsym dlsym "libdl.so.2"
//go:cgo_import_dynamic dlerror dlerror "libdl.so.2"
//go:cgo_import_dynamic _ _ "libdl.so.2"

// Available reports whether this build can call C.
const Available = true

const rtldNow = 2 // RTLD_NOW: resolve every symbol at load time

// The trampolines' addresses, set in dl_linux_amd64.s.
var (
	dlopenAddr         uintptr
	dlsymAddr          uintptr
	dlerrorAddr        uintptr
	callTrampolineAddr uintptr
)

// frame is the block callTrampoline reads its arguments from and writes its
// result to. The assembly depends on its layout.
type frame struct {
	fn   uintptr
	args [MaxArgs]uintptr
	r1   uintptr
}

// cgocall runs fn(arg) on the system stack, with the C calling convention,
// marking the goroutine as in a system call while it runs. The goroutine's
// stack cannot move until it returns, so arg may point into it.
//
//go:linkname cgocall runtime.cgocall
//go:noescape
func cgocall(fn uintptr, arg unsafe.Pointer) int32

// Call calls the C function at fn with args and returns its integer or
// pointer result.
//
//go:uintptrescapes
func Call(fn uintptr, args ...uintptr) uintptr {
	if len(args) > MaxArgs {
		panic("dl: Call supports at most 12 arguments")
	}
	f := frame{fn: fn}
	copy(f.args[:], args)
	cgocall(callTrampolineAddr, unsafe.Pointer(&f))
	return f.r1
}

// Open loads the shared library name, such as "libcuda.so.1", resolving
// its symbols now.
func Open(name string) (Lib, error) {
	runtime.LockOSThread() // dlerror is per-thread
	defer runtime.UnlockOSThread()
	b := cstring(name)
	h := Call(dlopenAddr, uintptr(unsafe.Pointer(&b[0])), rtldNow)
	if h == 0 {
		return 0, fmt.Errorf("dl: %s", lastError())
	}
	return Lib(h), nil
}

// Sym returns the address of the symbol name in l.
func (l Lib) Sym(name string) (uintptr, error) {
	runtime.LockOSThread() // dlerror is per-thread
	defer runtime.UnlockOSThread()
	b := cstring(name)
	Call(dlerrorAddr) // clear any earlier error
	p := Call(dlsymAddr, uintptr(l), uintptr(unsafe.Pointer(&b[0])))
	if p == 0 {
		return 0, fmt.Errorf("dl: %s", lastError())
	}
	return p, nil
}

// lastError returns dlerror's message, which is per-thread: the caller must
// have locked the OS thread across the failed call.
func lastError() string {
	if s := GoString(Call(dlerrorAddr)); s != "" {
		return s
	}
	return "unknown error"
}

// GoString copies the NUL-terminated C string at p.
func GoString(p uintptr) string {
	if p == 0 {
		return ""
	}
	s := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	n := 0
	for *(*byte)(unsafe.Add(s, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(s), n))
}
