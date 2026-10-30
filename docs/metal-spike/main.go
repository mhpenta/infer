//go:build darwin && arm64

package main

import (
	_ "embed"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

//go:embed kernels.metal
var src string

func init() { runtime.LockOSThread() } // autorelease pools belong to a thread

type mtlSize struct{ w, h, d uint64 }

func pageAlloc(n int) []byte {
	n = (n + 16383) &^ 16383
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	return b
}

func main() {
	pool := call(objc_autoreleasePoolPush_trampoline_addr)
	defer call(objc_autoreleasePoolPop_trampoline_addr, pool)

	dev := id(call(MTLCreateSystemDefaultDevice_trampoline_addr))
	if dev == 0 {
		panic("no Metal device")
	}
	queue := dev.send("newCommandQueue")
	t0 := time.Now()
	opts := class("MTLCompileOptions").send("new")
	opts.send("setFastMathEnabled:", 0)
	errp := new(id)
	lib := dev.send("newLibraryWithSource:options:error:", uintptr(nsString(src)), uintptr(opts), uintptr(unsafe.Pointer(errp)))
	if lib == 0 {
		panic(goString(errp.send("localizedDescription")))
	}
	pso := func(name string) id {
		f := lib.send("newFunctionWithName:", uintptr(nsString(name)))
		p := dev.send("newComputePipelineStateWithFunction:error:", uintptr(f), uintptr(unsafe.Pointer(errp)))
		if p == 0 {
			panic(name)
		}
		return p
	}
	gemvPSO, emptyPSO := pso("gemv_bf16"), pso("empty")
	fmt.Printf("device %s, CGO_ENABLED=0, runtime MSL compile %v\n", goString(dev.send("name")), time.Since(t0))

	wrap := func(b []byte) id {
		return dev.send("newBufferWithBytesNoCopy:length:options:deallocator:", uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), 0, 0)
	}
	sizes := (*[2]mtlSize)(unsafe.Pointer(&pageAlloc(64)[0])) // off the Go stack

	for _, sh := range [][2]int{{9216, 1536}, {130560, 1536}} {
		n, k := sh[0], sh[1]
		wb, xb, yb := pageAlloc(n*k*2), pageAlloc(k*4), pageAlloc(n*4)
		w := unsafe.Slice((*uint16)(unsafe.Pointer(&wb[0])), n*k)
		x := unsafe.Slice((*float32)(unsafe.Pointer(&xb[0])), k)
		y := unsafe.Slice((*float32)(unsafe.Pointer(&yb[0])), n)
		r := rand.New(rand.NewPCG(1, 2))
		for i := range w {
			w[i] = uint16(math.Float32bits(r.Float32()-0.5) >> 16)
		}
		for i := range x {
			x[i] = r.Float32() - 0.5
		}
		W, X, Y := wrap(wb), wrap(xb), wrap(yb)
		kp := (*int32)(unsafe.Pointer(&pageAlloc(16)[0]))
		*kp = int32(k)
		run := func(reps int) time.Duration {
			t := time.Now()
			cb := queue.send("commandBuffer")
			e := cb.send("computeCommandEncoder")
			e.send("setComputePipelineState:", uintptr(gemvPSO))
			e.send("setBuffer:offset:atIndex:", uintptr(W), 0, 0)
			e.send("setBuffer:offset:atIndex:", uintptr(X), 0, 1)
			e.send("setBuffer:offset:atIndex:", uintptr(Y), 0, 2)
			e.send("setBytes:length:atIndex:", uintptr(unsafe.Pointer(kp)), 4, 3)
			sizes[0], sizes[1] = mtlSize{uint64(n+31) / 32, 1, 1}, mtlSize{256, 1, 1}
			for range reps {
				e.send("dispatchThreadgroups:threadsPerThreadgroup:", uintptr(unsafe.Pointer(&sizes[0])), uintptr(unsafe.Pointer(&sizes[1])))
			}
			e.send("endEncoding")
			cb.send("commit")
			cb.send("waitUntilCompleted")
			return time.Since(t)
		}
		run(1)
		var maxErr float64
		for _, i := range []int{0, 1, n / 2, n - 1} {
			var s float64
			for j := range k {
				s += float64(math.Float32frombits(uint32(w[i*k+j])<<16)) * float64(x[j])
			}
			maxErr = max(maxErr, math.Abs(s-float64(y[i])))
		}
		reps := 50
		if n > 100000 {
			reps = 5
		}
		per := run(reps).Seconds() / float64(reps)
		fmt.Printf("gemv bf16 %dx%d: %.1f µs, %.0f GB/s (wall clock, incl. submit), max |err| vs float64 %.2g\n",
			n, k, per*1e6, float64(n*k*2)/per/1e9, maxErr)
	}

	one := (*[2]mtlSize)(unsafe.Pointer(&pageAlloc(64)[0]))
	*one = [2]mtlSize{{1, 1, 1}, {32, 1, 1}}
	trip := func() {
		p := call(objc_autoreleasePoolPush_trampoline_addr)
		cb := queue.send("commandBuffer")
		e := cb.send("computeCommandEncoder")
		e.send("setComputePipelineState:", uintptr(emptyPSO))
		e.send("dispatchThreadgroups:threadsPerThreadgroup:", uintptr(unsafe.Pointer(&one[0])), uintptr(unsafe.Pointer(&one[1])))
		e.send("endEncoding")
		cb.send("commit")
		cb.send("waitUntilCompleted")
		call(objc_autoreleasePoolPop_trampoline_addr, p)
	}
	for range 20 {
		trip()
	}
	const trips = 500
	t0 = time.Now()
	for range trips {
		trip()
	}
	fmt.Printf("empty commit+wait round trip: %.1f µs\n", float64(time.Since(t0).Microseconds())/trips)
	t0 = time.Now()
	for range 100000 {
		queue.send("label")
	}
	fmt.Printf("one objc_msgSend from Go: %.0f ns\n", float64(time.Since(t0).Nanoseconds())/100000)
}
