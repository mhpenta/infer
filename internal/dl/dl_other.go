//go:build !linux || !amd64 || cgo

package dl

// Available reports whether this build can call C.
const Available = false

// Call returns 0: this build cannot call C.
func Call(fn uintptr, args ...uintptr) uintptr { return 0 }

// Open returns ErrUnavailable.
func Open(name string) (Lib, error) { return 0, ErrUnavailable }

// Sym returns ErrUnavailable.
func (l Lib) Sym(name string) (uintptr, error) { return 0, ErrUnavailable }

// GoString returns "".
func GoString(p uintptr) string { return "" }
