// Package dl loads shared libraries and calls their C functions without cgo,
// so package cuda can drive libcuda from pure Go on Linux.
//
// Open and Sym wrap glibc's dlopen and dlsym, reached through
// //go:cgo_import_dynamic. Call runs a C function on the system stack
// through runtime.cgocall, as cgo-generated code does, so a blocking call
// does not hold up the scheduler. Package internal/fakecgo makes that safe:
// it has the runtime start every thread with pthread_create, so glibc's
// thread-local storage is set up wherever the call lands.
//
// Only linux/amd64 builds with cgo disabled have it; elsewhere Available is
// false and Open fails. Arguments and results are integers or pointers,
// passed as uintptr. Float arguments and results are not supported.
//
// Pass a Go pointer as uintptr(unsafe.Pointer(p)) written in the call's
// argument list. Call is marked //go:uintptrescapes, so the compiler moves
// what p points to onto the heap and keeps it alive until the call returns.
// Go memory that C reads through a pointer stored in other memory, such as
// an array of argument pointers, must be pinned with runtime.Pinner.
package dl

import "errors"

// MaxArgs is the most arguments Call passes.
const MaxArgs = 12

// ErrUnavailable is returned by Open when this build cannot call C.
var ErrUnavailable = errors.New("dl: calling C needs linux/amd64 with CGO_ENABLED=0")

// Lib is a handle to a loaded shared library.
type Lib uintptr

func cstring(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}
