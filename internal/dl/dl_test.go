package dl

import (
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

func openLibc(t *testing.T) Lib {
	t.Helper()
	if !Available {
		t.Skip("dl needs linux/amd64 with CGO_ENABLED=0")
	}
	l, err := Open("libc.so.6")
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func sym(t *testing.T, l Lib, name string) uintptr {
	t.Helper()
	p, err := l.Sym(name)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCall(t *testing.T) {
	l := openLibc(t)
	strlen, getpid := sym(t, l, "strlen"), sym(t, l, "getpid")

	s := []byte("hello, libc\x00")
	if n := Call(strlen, uintptr(unsafe.Pointer(&s[0]))); n != 11 {
		t.Errorf("strlen = %d, want 11", n)
	}
	if pid := int(int32(Call(getpid))); pid != os.Getpid() {
		t.Errorf("getpid = %d, want %d", pid, os.Getpid())
	}
}

// TestCallManyArgs passes all 12 arguments, six of them on the stack,
// through snprintf, and checks every one arrives.
func TestCallManyArgs(t *testing.T) {
	l := openLibc(t)
	snprintf := sym(t, l, "snprintf")
	buf := make([]byte, 128)
	format := []byte("%ld %ld %ld %ld %ld %ld %ld %ld %ld\x00")
	n := Call(snprintf, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), uintptr(unsafe.Pointer(&format[0])),
		1, 2, 3, 4, 5, 6, 7, 8, ^uintptr(8)) // the last is -9
	want := "1 2 3 4 5 6 7 8 -9"
	if got := string(buf[:n]); got != want {
		t.Errorf("snprintf wrote %q, want %q", got, want)
	}
}

// TestCallConcurrent calls C from many goroutines on threads the runtime
// starts for them, which fakecgo creates with pthread_create. A thread with
// no glibc TLS would crash in strerror, which uses it.
func TestCallConcurrent(t *testing.T) {
	l := openLibc(t)
	strerror, strlen := sym(t, l, "strerror"), sym(t, l, "strlen")
	var wg sync.WaitGroup
	errs := make(chan string, 64)
	for range 64 {
		wg.Go(func() {
			runtime.LockOSThread() // force the runtime to start more threads
			defer runtime.UnlockOSThread()
			for range 100 {
				msg := GoString(Call(strerror, 2)) // ENOENT
				if !strings.Contains(msg, "No such file") {
					errs <- msg
					return
				}
				Call(strlen, Call(strerror, 2))
			}
		})
	}
	wg.Wait()
	close(errs)
	for msg := range errs {
		t.Errorf("strerror(ENOENT) = %q", msg)
	}
}

func TestSetenvReachesC(t *testing.T) {
	l := openLibc(t)
	getenv := sym(t, l, "getenv")
	t.Setenv("INFER_DL_TEST", "visible")
	k := []byte("INFER_DL_TEST\x00")
	if got := GoString(Call(getenv, uintptr(unsafe.Pointer(&k[0])))); got != "visible" {
		t.Errorf("C getenv = %q, want %q", got, "visible")
	}
}

func TestOpenErrors(t *testing.T) {
	l := openLibc(t)
	if _, err := Open("libdoes-not-exist.so"); err == nil || !strings.Contains(err.Error(), "libdoes-not-exist.so") {
		t.Errorf("Open of a missing library: err = %v", err)
	}
	if _, err := l.Sym("no_such_symbol_here"); err == nil || !strings.Contains(err.Error(), "no_such_symbol_here") {
		t.Errorf("Sym of a missing symbol: err = %v", err)
	}
}
