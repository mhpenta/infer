// Package cuda runs compute kernels on NVIDIA GPUs through the CUDA driver
// API, from pure Go: it loads libcuda.so.1 and calls it with internal/dl,
// without cgo. Kernels are PTX, which the driver compiles for the GPU at
// load time (Device.LoadPTX), so building needs no CUDA toolkit; running
// needs only the NVIDIA driver.
//
// This is the bridge: devices, device memory, copies, PTX modules and kernel
// launches. Only linux/amd64 builds with CGO_ENABLED=0 have it; elsewhere,
// or with no driver or GPU, Open returns an error wrapping ErrUnavailable.
//
// A CUDA context is current per OS thread, and goroutines move between
// threads, so every Device method locks its goroutine to a thread and makes
// the device's context current for the length of the call. A Device is safe
// for concurrent use.
package cuda

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/mhpenta/infer/internal/dl"
)

// ErrUnavailable is wrapped by the error Open returns when there is no CUDA
// driver or no GPU.
var ErrUnavailable = errors.New("cuda: no CUDA device")

// Error is a CUDA driver API failure.
type Error struct {
	Op   string // the driver function, such as "cuMemAlloc_v2"
	Code int    // the CUresult
	Name string // the CUresult's name, such as "CUDA_ERROR_OUT_OF_MEMORY"
	Msg  string // the driver's description
}

func (e *Error) Error() string {
	return fmt.Sprintf("cuda: %s: %s (%s)", e.Op, e.Msg, e.Name)
}

// The driver functions, resolved once by load. Versioned names (_v2) are
// the current ABI of functions whose signatures changed.
var driver struct {
	init, driverGetVersion, getErrorName, getErrorString                   uintptr
	deviceGetCount, deviceGet, deviceGetName, deviceGetAttribute, totalMem uintptr
	primaryCtxRetain, primaryCtxRelease, ctxSetCurrent, ctxSynchronize     uintptr
	memAlloc, memFree, memcpyHtoD, memcpyDtoH                              uintptr
	moduleLoadDataEx, moduleUnload, moduleGetFunction, launchKernel        uintptr
}

var (
	loadOnce sync.Once
	loadErr  error
)

func load() error {
	loadOnce.Do(func() {
		if !dl.Available {
			loadErr = fmt.Errorf("%w: %w", ErrUnavailable, dl.ErrUnavailable)
			return
		}
		lib, err := dl.Open("libcuda.so.1")
		if err != nil {
			loadErr = fmt.Errorf("%w: %w", ErrUnavailable, err)
			return
		}
		syms := []struct {
			p    *uintptr
			name string
		}{
			{&driver.init, "cuInit"},
			{&driver.driverGetVersion, "cuDriverGetVersion"},
			{&driver.getErrorName, "cuGetErrorName"},
			{&driver.getErrorString, "cuGetErrorString"},
			{&driver.deviceGetCount, "cuDeviceGetCount"},
			{&driver.deviceGet, "cuDeviceGet"},
			{&driver.deviceGetName, "cuDeviceGetName"},
			{&driver.deviceGetAttribute, "cuDeviceGetAttribute"},
			{&driver.totalMem, "cuDeviceTotalMem_v2"},
			{&driver.primaryCtxRetain, "cuDevicePrimaryCtxRetain"},
			{&driver.primaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"},
			{&driver.ctxSetCurrent, "cuCtxSetCurrent"},
			{&driver.ctxSynchronize, "cuCtxSynchronize"},
			{&driver.memAlloc, "cuMemAlloc_v2"},
			{&driver.memFree, "cuMemFree_v2"},
			{&driver.memcpyHtoD, "cuMemcpyHtoD_v2"},
			{&driver.memcpyDtoH, "cuMemcpyDtoH_v2"},
			{&driver.moduleLoadDataEx, "cuModuleLoadDataEx"},
			{&driver.moduleUnload, "cuModuleUnload"},
			{&driver.moduleGetFunction, "cuModuleGetFunction"},
			{&driver.launchKernel, "cuLaunchKernel"},
		}
		for _, s := range syms {
			if *s.p, err = lib.Sym(s.name); err != nil {
				loadErr = fmt.Errorf("cuda: libcuda.so.1 is too old: %w", err)
				return
			}
		}
		loadErr = check("cuInit", dl.Call(driver.init, 0))
	})
	return loadErr
}

// check turns a CUresult into an error.
func check(op string, r uintptr) error {
	code := int(int32(r))
	if code == 0 {
		return nil
	}
	e := &Error{Op: op, Code: code, Name: fmt.Sprintf("CUresult %d", code), Msg: "unknown error"}
	var s uintptr
	if int32(dl.Call(driver.getErrorName, r, uintptr(unsafe.Pointer(&s)))) == 0 {
		e.Name = dl.GoString(s)
	}
	if int32(dl.Call(driver.getErrorString, r, uintptr(unsafe.Pointer(&s)))) == 0 {
		e.Msg = dl.GoString(s)
	}
	return e
}

// DriverVersion returns the CUDA version the driver supports, such as
// 12090 for 12.9.
func DriverVersion() (int, error) {
	if err := load(); err != nil {
		return 0, err
	}
	var v int32
	err := check("cuDriverGetVersion", dl.Call(driver.driverGetVersion, uintptr(unsafe.Pointer(&v))))
	return int(v), err
}

// Device is a GPU and its primary context.
type Device struct {
	dev      int32
	ctx      uintptr
	name     string
	cc       [2]int
	totalMem int64
}

// Device attributes, from CUdevice_attribute.
const (
	attrComputeCapabilityMajor = 75
	attrComputeCapabilityMinor = 76
)

// Open returns GPU number ordinal, counting from 0.
func Open(ordinal int) (*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	var n int32
	if err := check("cuDeviceGetCount", dl.Call(driver.deviceGetCount, uintptr(unsafe.Pointer(&n)))); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnavailable, err)
	}
	if ordinal < 0 || ordinal >= int(n) {
		return nil, fmt.Errorf("%w: device %d requested, %d present", ErrUnavailable, ordinal, n)
	}
	d := &Device{}
	if err := check("cuDeviceGet", dl.Call(driver.deviceGet, uintptr(unsafe.Pointer(&d.dev)), uintptr(ordinal))); err != nil {
		return nil, err
	}
	name := make([]byte, 256)
	if err := check("cuDeviceGetName", dl.Call(driver.deviceGetName, uintptr(unsafe.Pointer(&name[0])), uintptr(len(name)), uintptr(d.dev))); err != nil {
		return nil, err
	}
	d.name = dl.GoString(uintptr(unsafe.Pointer(&name[0])))
	for i, attr := range []uintptr{attrComputeCapabilityMajor, attrComputeCapabilityMinor} {
		var v int32
		if err := check("cuDeviceGetAttribute", dl.Call(driver.deviceGetAttribute, uintptr(unsafe.Pointer(&v)), attr, uintptr(d.dev))); err != nil {
			return nil, err
		}
		d.cc[i] = int(v)
	}
	var total uintptr
	if err := check("cuDeviceTotalMem_v2", dl.Call(driver.totalMem, uintptr(unsafe.Pointer(&total)), uintptr(d.dev))); err != nil {
		return nil, err
	}
	d.totalMem = int64(total)
	if err := check("cuDevicePrimaryCtxRetain", dl.Call(driver.primaryCtxRetain, uintptr(unsafe.Pointer(&d.ctx)), uintptr(d.dev))); err != nil {
		return nil, err
	}
	return d, nil
}

// Close releases the device's primary context. Everything allocated or
// loaded on the device must be freed first.
func (d *Device) Close() error {
	if d.ctx == 0 {
		return nil
	}
	d.ctx = 0
	return check("cuDevicePrimaryCtxRelease_v2", dl.Call(driver.primaryCtxRelease, uintptr(d.dev)))
}

// Name returns the GPU's name, such as "NVIDIA L4".
func (d *Device) Name() string { return d.name }

// ComputeCapability returns the GPU's compute capability, such as 8, 9 for
// an L4.
func (d *Device) ComputeCapability() (major, minor int) { return d.cc[0], d.cc[1] }

// TotalMemory returns the GPU's memory in bytes.
func (d *Device) TotalMemory() int64 { return d.totalMem }

// call runs the driver function fn with d's context current on this thread.
//
//go:uintptrescapes
func (d *Device) call(op string, fn uintptr, args ...uintptr) error {
	if d.ctx == 0 {
		panic("cuda: use of a closed Device")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := check("cuCtxSetCurrent", dl.Call(driver.ctxSetCurrent, d.ctx)); err != nil {
		return err
	}
	return check(op, dl.Call(fn, args...))
}

// Synchronize waits for all work on the device to finish, and returns the
// first error a kernel hit.
func (d *Device) Synchronize() error {
	return d.call("cuCtxSynchronize", driver.ctxSynchronize)
}

// Buffer is device memory.
type Buffer struct {
	d    *Device
	ptr  uintptr // CUdeviceptr
	size int
}

// Alloc allocates size bytes of device memory.
func (d *Device) Alloc(size int) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("cuda: Alloc of %d bytes", size)
	}
	b := &Buffer{d: d, size: size}
	if err := d.call("cuMemAlloc_v2", driver.memAlloc, uintptr(unsafe.Pointer(&b.ptr)), uintptr(size)); err != nil {
		return nil, err
	}
	return b, nil
}

// Free releases the buffer's memory.
func (b *Buffer) Free() error {
	if b.ptr == 0 {
		return nil
	}
	p := b.ptr
	b.ptr = 0
	return b.d.call("cuMemFree_v2", driver.memFree, p)
}

// Len returns the buffer's size in bytes.
func (b *Buffer) Len() int { return b.size }

// Write copies src to the start of b. It returns once the copy is done.
func Write[T any](b *Buffer, src []T) error {
	n := len(src) * int(unsafe.Sizeof(*new(T)))
	if n > b.size {
		return fmt.Errorf("cuda: Write of %d bytes to a %d-byte buffer", n, b.size)
	}
	if n == 0 {
		return nil
	}
	return b.d.call("cuMemcpyHtoD_v2", driver.memcpyHtoD, b.ptr, uintptr(unsafe.Pointer(&src[0])), uintptr(n))
}

// Read copies the start of b to dst. It waits for earlier work on the
// device to finish first.
func Read[T any](b *Buffer, dst []T) error {
	n := len(dst) * int(unsafe.Sizeof(*new(T)))
	if n > b.size {
		return fmt.Errorf("cuda: Read of %d bytes from a %d-byte buffer", n, b.size)
	}
	if n == 0 {
		return nil
	}
	return b.d.call("cuMemcpyDtoH_v2", driver.memcpyDtoH, uintptr(unsafe.Pointer(&dst[0])), b.ptr, uintptr(n))
}

// Module is loaded GPU code.
type Module struct {
	d   *Device
	mod uintptr // CUmodule
}

// JIT options, from CUjit_option.
const (
	jitErrorLogBuffer          = 5
	jitErrorLogBufferSizeBytes = 6
)

// LoadPTX compiles PTX source for the device and loads it. A compile error
// includes the driver's log.
func (d *Device) LoadPTX(ptx string) (*Module, error) {
	src := make([]byte, len(ptx)+1)
	copy(src, ptx)
	log := make([]byte, 8192)
	opts := []int32{jitErrorLogBuffer, jitErrorLogBufferSizeBytes}
	vals := []uintptr{uintptr(unsafe.Pointer(&log[0])), uintptr(len(log))}
	var pin runtime.Pinner // vals holds log's address where the GC cannot see it
	pin.Pin(&log[0])
	defer pin.Unpin()

	m := &Module{d: d}
	err := d.call("cuModuleLoadDataEx", driver.moduleLoadDataEx, uintptr(unsafe.Pointer(&m.mod)),
		uintptr(unsafe.Pointer(&src[0])), uintptr(len(opts)), uintptr(unsafe.Pointer(&opts[0])), uintptr(unsafe.Pointer(&vals[0])))
	if err != nil {
		if s := dl.GoString(uintptr(unsafe.Pointer(&log[0]))); s != "" {
			err = fmt.Errorf("%w:\n%s", err, s)
		}
		return nil, err
	}
	return m, nil
}

// Unload frees the module. Its Functions must not be launched afterwards.
func (m *Module) Unload() error {
	if m.mod == 0 {
		return nil
	}
	p := m.mod
	m.mod = 0
	return m.d.call("cuModuleUnload", driver.moduleUnload, p)
}

// Function is a kernel in a Module.
type Function struct {
	m    *Module
	fn   uintptr // CUfunction
	name string
}

// Function returns the kernel named name: the name of a .entry in the PTX.
func (m *Module) Function(name string) (*Function, error) {
	cname := make([]byte, len(name)+1)
	copy(cname, name)
	f := &Function{m: m, name: name}
	if err := m.d.call("cuModuleGetFunction", driver.moduleGetFunction, uintptr(unsafe.Pointer(&f.fn)), m.mod, uintptr(unsafe.Pointer(&cname[0]))); err != nil {
		return nil, fmt.Errorf("%w: kernel %q", err, name)
	}
	return f, nil
}

// Dim3 is a grid or block size: X, Y, Z, each at least 1.
type Dim3 struct{ X, Y, Z uint32 }

// Launch queues f on grid blocks of block threads, with sharedMem bytes of
// dynamic shared memory per block, and returns without waiting; Read and
// Synchronize wait. Each argument fills the kernel parameter at its
// position and must have that parameter's type: a *Buffer for a .u64
// pointer, or an int32, uint32, float32, int64, uint64 or float64. Another
// type panics.
func (f *Function) Launch(grid, block Dim3, sharedMem int, args ...any) error {
	// cuLaunchKernel takes an array of pointers, one to each argument's
	// value. Each value gets an 8-byte slot; a 4-byte parameter reads the low
	// half, which on little-endian amd64 is where it was stored.
	vals := make([]uint64, len(args)+1) // +1: never empty
	ptrs := make([]uintptr, len(args)+1)
	var pin runtime.Pinner // ptrs holds vals' addresses where the GC cannot see them
	pin.Pin(&vals[0])
	defer pin.Unpin()
	for i, a := range args {
		switch a := a.(type) {
		case *Buffer:
			vals[i] = uint64(a.ptr)
		case int32:
			vals[i] = uint64(uint32(a))
		case uint32:
			vals[i] = uint64(a)
		case float32:
			*(*float32)(unsafe.Pointer(&vals[i])) = a
		case int64:
			vals[i] = uint64(a)
		case uint64:
			vals[i] = a
		case float64:
			*(*float64)(unsafe.Pointer(&vals[i])) = a
		default:
			panic(fmt.Sprintf("cuda: kernel %s argument %d has unsupported type %T", f.name, i, a))
		}
		ptrs[i] = uintptr(unsafe.Pointer(&vals[i]))
	}
	return f.m.d.call("cuLaunchKernel", driver.launchKernel, f.fn,
		uintptr(grid.X), uintptr(grid.Y), uintptr(grid.Z),
		uintptr(block.X), uintptr(block.Y), uintptr(block.Z),
		uintptr(sharedMem), 0, // the default stream
		uintptr(unsafe.Pointer(&ptrs[0])), 0)
}
