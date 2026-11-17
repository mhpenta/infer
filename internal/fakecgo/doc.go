// Package fakecgo lets a program built with CGO_ENABLED=0 call C libraries
// on Linux. Importing it for its side effects is all it takes.
//
// Without cgo, Go's runtime on Linux starts threads with a raw clone system
// call and points the FS register at its own thread-local storage. C code
// compiled against glibc expects every thread to have been made by
// pthread_create and FS to point at glibc's thread control block, so calling
// into libcuda from such a thread crashes. Package runtime/cgo fixes this by
// filling in hooks the runtime calls when runtime.iscgo is true: _cgo_init
// at startup, before Go touches TLS, and _cgo_thread_start for every new
// thread. This package fills in the same hooks without a C compiler, in
// assembly that calls glibc through //go:cgo_import_dynamic, the way purego's
// "fakecgo" does. The main thread keeps the TLS glibc's loader gave it, and
// every other thread comes from pthread_create.
//
// The hooks need a program that does not otherwise link runtime/cgo, so the
// package is empty unless the build is linux/amd64 with cgo disabled.
// Callbacks from C into Go are not supported.
package fakecgo
