//go:build !cgo

package fakecgo

import _ "unsafe" // for go:linkname

// The glibc functions the hooks in fakecgo_linux_amd64.s call. libpthread
// and libdl are stubs in glibc 2.34 and later, but naming them keeps older
// glibc working.
//
//go:cgo_import_dynamic malloc malloc "libc.so.6"
//go:cgo_import_dynamic free free "libc.so.6"
//go:cgo_import_dynamic abort abort "libc.so.6"
//go:cgo_import_dynamic setenv setenv "libc.so.6"
//go:cgo_import_dynamic unsetenv unsetenv "libc.so.6"
//go:cgo_import_dynamic sigfillset sigfillset "libc.so.6"
//go:cgo_import_dynamic pthread_sigmask pthread_sigmask "libpthread.so.0"
//go:cgo_import_dynamic pthread_attr_init pthread_attr_init "libpthread.so.0"
//go:cgo_import_dynamic pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so.0"
//go:cgo_import_dynamic pthread_attr_destroy pthread_attr_destroy "libpthread.so.0"
//go:cgo_import_dynamic pthread_create pthread_create "libpthread.so.0"
//go:cgo_import_dynamic pthread_detach pthread_detach "libpthread.so.0"
//go:cgo_import_dynamic _ _ "libc.so.6"
//go:cgo_import_dynamic _ _ "libpthread.so.0"

// The runtime checks iscgo and set_crosscall2 before any package's init
// runs, so both are set statically. The C function pointers the runtime
// reads (_cgo_init, _cgo_thread_start and the rest) are defined in
// fakecgo_linux_amd64.s.

//go:linkname _iscgo runtime.iscgo
var _iscgo = true

//go:linkname _set_crosscall2 runtime.set_crosscall2
var _set_crosscall2 = setCrosscall2

// setCrosscall2 would let C call back into Go. Callbacks are not supported,
// so there is nothing to set.
func setCrosscall2() {}
