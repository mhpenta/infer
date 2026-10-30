// Package metal runs compute kernels on Apple silicon GPUs through Metal,
// from pure Go: it sends Objective-C messages with internal/objc, without
// cgo. Kernels are Metal Shading Language source compiled at run time by the
// OS (Device.Compile), so building needs no Xcode or SDK.
//
// The wrapper is thin. A Device compiles Libraries, whose kernels become
// Pipelines; Buffers hold data in memory the CPU and GPU share; a
// CommandBuffer records dispatches and runs them all with one Wait. A round
// trip to the GPU costs about 200 µs whatever the work, so a caller records a
// whole forward pass into one CommandBuffer rather than waiting per kernel.
//
// Only darwin/arm64 builds have Metal; elsewhere Open returns
// ErrUnavailable. Misuse (a buffer index out of range, a dispatch with no
// pipeline) panics, because it is a programming error and Metal itself would
// abort the process.
package metal

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/mhpenta/infer/internal/objc"
)

// ErrUnavailable is returned by Open when there is no Metal device.
var ErrUnavailable = errors.New("metal: no Metal device")

var (
	selNewCommandQueue       = objc.RegisterName("newCommandQueue")
	selName                  = objc.RegisterName("name")
	selHasUnifiedMemory      = objc.RegisterName("hasUnifiedMemory")
	selMaxBufferLength       = objc.RegisterName("maxBufferLength")
	selRecommendedWorkingSet = objc.RegisterName("recommendedMaxWorkingSetSize")
	selNew                   = objc.RegisterName("new")
	selSetFastMathEnabled    = objc.RegisterName("setFastMathEnabled:")
	selNewLibraryWithSource  = objc.RegisterName("newLibraryWithSource:options:error:")
	selNewFunctionWithName   = objc.RegisterName("newFunctionWithName:")
	selNewPipeline           = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selMaxThreads            = objc.RegisterName("maxTotalThreadsPerThreadgroup")
	selExecutionWidth        = objc.RegisterName("threadExecutionWidth")
	selNewBuffer             = objc.RegisterName("newBufferWithLength:options:")
	selNewBufferNoCopy       = objc.RegisterName("newBufferWithBytesNoCopy:length:options:deallocator:")
	selContents              = objc.RegisterName("contents")
	selCommandBuffer         = objc.RegisterName("commandBuffer")
	selComputeEncoder        = objc.RegisterName("computeCommandEncoder")
	selSetPipeline           = objc.RegisterName("setComputePipelineState:")
	selSetBuffer             = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBytes              = objc.RegisterName("setBytes:length:atIndex:")
	selDispatchThreadgroups  = objc.RegisterName("dispatchThreadgroups:threadsPerThreadgroup:")
	selDispatchThreads       = objc.RegisterName("dispatchThreads:threadsPerThreadgroup:")
	selEndEncoding           = objc.RegisterName("endEncoding")
	selCommit                = objc.RegisterName("commit")
	selWaitUntilCompleted    = objc.RegisterName("waitUntilCompleted")
	selStatus                = objc.RegisterName("status")
	selError                 = objc.RegisterName("error")

	classCompileOptions = objc.GetClass("MTLCompileOptions")
)

const (
	storageModeShared        = 0 // MTLResourceStorageModeShared: CPU and GPU see the same memory
	commandBufferStatusError = 5 // MTLCommandBufferStatusError
	maxBufferIndex           = 30
	maxSetBytes              = 4096 // Metal's limit for setBytes:length:atIndex:
)

// Device is a GPU with a command queue. It is safe for concurrent use; the
// CommandBuffers it creates are not.
type Device struct {
	id, queue objc.ID
	name      string
}

// Open returns the system's default Metal device, or ErrUnavailable.
func Open() (*Device, error) {
	if !objc.Available {
		return nil, ErrUnavailable
	}
	pool := objc.PushPool()
	defer pool.Pop()
	dev := objc.MTLCreateSystemDefaultDevice()
	if dev == 0 {
		return nil, ErrUnavailable
	}
	queue := objc.ID(dev.Send(selNewCommandQueue))
	if queue == 0 {
		dev.Release()
		return nil, errors.New("metal: cannot create a command queue")
	}
	return &Device{id: dev, queue: queue, name: objc.ID(dev.Send(selName)).String()}, nil
}

// Close releases the device. Everything made from it must be released first.
func (d *Device) Close() {
	d.queue.Release()
	d.id.Release()
	d.queue, d.id = 0, 0
}

// Name returns the GPU's name, such as "Apple M2 Max".
func (d *Device) Name() string { return d.name }

// UnifiedMemory reports whether the CPU and GPU share memory, as on all
// Apple silicon.
func (d *Device) UnifiedMemory() bool { return d.id.SendBool(selHasUnifiedMemory) }

// MaxBufferLength returns the largest Buffer the device can allocate, in
// bytes.
func (d *Device) MaxBufferLength() int { return int(d.id.Send(selMaxBufferLength)) }

// RecommendedMaxWorkingSetSize returns how many bytes of buffers the GPU can
// use without hurting performance.
func (d *Device) RecommendedMaxWorkingSetSize() int64 {
	return int64(d.id.Send(selRecommendedWorkingSet))
}

// Library is compiled kernel source.
type Library struct {
	dev *Device
	id  objc.ID
}

// Compile compiles Metal Shading Language source, with fast math off so
// results follow IEEE 754 float32 as the CPU kernels do.
func (d *Device) Compile(src string) (*Library, error) {
	pool := objc.PushPool()
	defer pool.Pop()
	opts := objc.ID(classCompileOptions.Send(selNew))
	defer opts.Release()
	opts.Send(selSetFastMathEnabled, 0)
	str := objc.NewString(src)
	defer str.Release()
	var errp objc.ID // set to an autoreleased NSError on failure
	lib := objc.ID(d.id.Send(selNewLibraryWithSource, uintptr(str), uintptr(opts), uintptr(unsafe.Pointer(&errp))))
	if lib == 0 {
		return nil, fmt.Errorf("metal: compiling: %s", errp.Description())
	}
	return &Library{dev: d, id: lib}, nil
}

// Release releases the library. Pipelines made from it stay valid.
func (l *Library) Release() {
	l.id.Release()
	l.id = 0
}

// Pipeline is a kernel ready to dispatch.
type Pipeline struct {
	id         objc.ID
	name       string
	maxThreads int
	width      int
}

// Pipeline returns the kernel function name, compiled for the device.
func (l *Library) Pipeline(name string) (*Pipeline, error) {
	pool := objc.PushPool()
	defer pool.Pop()
	str := objc.NewString(name)
	defer str.Release()
	fn := objc.ID(l.id.Send(selNewFunctionWithName, uintptr(str)))
	if fn == 0 {
		return nil, fmt.Errorf("metal: no kernel %q in the library", name)
	}
	defer fn.Release()
	var errp objc.ID
	pso := objc.ID(l.dev.id.Send(selNewPipeline, uintptr(fn), uintptr(unsafe.Pointer(&errp))))
	if pso == 0 {
		return nil, fmt.Errorf("metal: kernel %q: %s", name, errp.Description())
	}
	return &Pipeline{
		id:         pso,
		name:       name,
		maxThreads: int(pso.Send(selMaxThreads)),
		width:      int(pso.Send(selExecutionWidth)),
	}, nil
}

// MaxThreadsPerThreadgroup returns the largest threadgroup the kernel can
// run in, which depends on its register use.
func (p *Pipeline) MaxThreadsPerThreadgroup() int { return p.maxThreads }

// ThreadExecutionWidth returns the number of threads that execute in
// lockstep: the simdgroup width, 32 on Apple GPUs.
func (p *Pipeline) ThreadExecutionWidth() int { return p.width }

// Release releases the pipeline.
func (p *Pipeline) Release() {
	p.id.Release()
	p.id = 0
}

// Buffer is memory shared by the CPU and GPU. The CPU may read and write
// Bytes while no CommandBuffer using the buffer is running.
type Buffer struct {
	id   objc.ID
	data []byte
}

// NewBuffer allocates an n-byte buffer. Its initial contents are
// unspecified.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	if n <= 0 {
		panic(fmt.Sprintf("metal.NewBuffer: %d bytes", n))
	}
	id := objc.ID(d.id.Send(selNewBuffer, uintptr(n), storageModeShared))
	if id == 0 {
		return nil, fmt.Errorf("metal: cannot allocate a %d-byte buffer", n)
	}
	return &Buffer{id: id, data: unsafe.Slice((*byte)(objc.Pointer(id.Send(selContents))), n)}, nil
}

// Wrap returns a buffer over b without copying it. b must be memory outside
// the Go heap, such as an mmap'd file or anonymous mapping, whose start and
// length are multiples of the page size; it must stay mapped until the
// buffer is released.
func (d *Device) Wrap(b []byte) (*Buffer, error) {
	page := os.Getpagesize()
	if len(b) == 0 || uintptr(unsafe.Pointer(&b[0]))%uintptr(page) != 0 || len(b)%page != 0 {
		return nil, fmt.Errorf("metal: Wrap needs page-aligned memory (%d-byte pages), got %d bytes", page, len(b))
	}
	id := objc.ID(d.id.Send(selNewBufferNoCopy, uintptr(unsafe.Pointer(&b[0])), uintptr(len(b)), storageModeShared, 0))
	if id == 0 {
		return nil, fmt.Errorf("metal: cannot wrap %d bytes", len(b))
	}
	return &Buffer{id: id, data: b}, nil
}

// Bytes returns the buffer's memory.
func (b *Buffer) Bytes() []byte { return b.data }

// Len returns the buffer's size in bytes.
func (b *Buffer) Len() int { return len(b.data) }

// Release releases the buffer. A wrapped buffer's memory stays with its
// owner.
func (b *Buffer) Release() {
	b.id.Release()
	b.id, b.data = 0, nil
}

// Slice returns a buffer's memory as a slice of T, dropping any bytes past
// the last whole element.
func Slice[T float32 | uint16 | int8 | int32 | uint32](b *Buffer) []T {
	var zero T
	n := len(b.data) / int(unsafe.Sizeof(zero))
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&b.data[0])), n)
}

// Size is a grid or threadgroup size, laid out as MTLSize.
type Size struct{ X, Y, Z int }

// CommandBuffer records kernel dispatches, which run in order, each seeing
// the writes of the ones before it, when Wait is called. It belongs to the
// goroutine that created it: from NewCommandBuffer to Wait, that goroutine
// is locked to its OS thread, which holds the autorelease pool for the
// command buffer's Metal objects.
type CommandBuffer struct {
	pool     objc.Pool
	cb, enc  objc.ID
	pipeline *Pipeline
	done     bool
}

// NewCommandBuffer starts recording. Every CommandBuffer must end with Wait.
func (d *Device) NewCommandBuffer() *CommandBuffer {
	pool := objc.PushPool()
	cb := objc.ID(d.queue.Send(selCommandBuffer)) // autoreleased into pool
	enc := objc.ID(cb.Send(selComputeEncoder))
	return &CommandBuffer{pool: pool, cb: cb, enc: enc}
}

func (c *CommandBuffer) check(name string) {
	if c.done {
		panic("metal.CommandBuffer." + name + ": called after Wait")
	}
}

// SetPipeline sets the kernel for the following dispatches.
func (c *CommandBuffer) SetPipeline(p *Pipeline) {
	c.check("SetPipeline")
	c.enc.Send(selSetPipeline, uintptr(p.id))
	c.pipeline = p
}

// SetBuffer binds b, from byte offset on, to the kernel's [[buffer(index)]].
func (c *CommandBuffer) SetBuffer(index int, b *Buffer, offset int) {
	c.check("SetBuffer")
	if index < 0 || index > maxBufferIndex || offset < 0 || offset >= b.Len() {
		panic(fmt.Sprintf("metal.CommandBuffer.SetBuffer: index %d, offset %d into %d bytes", index, offset, b.Len()))
	}
	c.enc.Send(selSetBuffer, uintptr(b.id), uintptr(offset), uintptr(index))
}

// SetBytes copies data, at most 4 KB, into the command buffer and binds it to
// the kernel's [[buffer(index)]]: the way to pass small arguments.
func (c *CommandBuffer) SetBytes(index int, data []byte) {
	c.check("SetBytes")
	if index < 0 || index > maxBufferIndex || len(data) == 0 || len(data) > maxSetBytes {
		panic(fmt.Sprintf("metal.CommandBuffer.SetBytes: index %d, %d bytes", index, len(data)))
	}
	c.enc.Send(selSetBytes, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), uintptr(index))
}

// SetValue binds a copy of v, such as a struct of kernel parameters, to the
// kernel's [[buffer(index)]]. v's layout must match the kernel's type.
func SetValue[T any](c *CommandBuffer, index int, v T) {
	c.SetBytes(index, unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v)))
}

// DispatchThreadgroups runs the pipeline on a grid of groups threadgroups,
// each of threads threads.
func (c *CommandBuffer) DispatchThreadgroups(groups, threads Size) {
	c.dispatch("DispatchThreadgroups", selDispatchThreadgroups, groups, threads)
}

// DispatchThreads runs the pipeline on a grid of threads threads, cut into
// threadgroups of perGroup; the last group in each dimension may be partial.
func (c *CommandBuffer) DispatchThreads(threads, perGroup Size) {
	c.dispatch("DispatchThreads", selDispatchThreads, threads, perGroup)
}

func (c *CommandBuffer) dispatch(name string, sel objc.SEL, grid, group Size) {
	c.check(name)
	if c.pipeline == nil {
		panic("metal.CommandBuffer." + name + ": no pipeline set")
	}
	if grid.X <= 0 || grid.Y <= 0 || grid.Z <= 0 || group.X <= 0 || group.Y <= 0 || group.Z <= 0 ||
		group.X*group.Y*group.Z > c.pipeline.maxThreads {
		panic(fmt.Sprintf("metal.CommandBuffer.%s: grid %v, threadgroup %v; %s allows %d threads per group",
			name, grid, group, c.pipeline.name, c.pipeline.maxThreads))
	}
	// MTLSize is three NSUIntegers, 24 bytes, so arm64 passes each by
	// pointer to a copy.
	c.enc.Send(sel, uintptr(unsafe.Pointer(&grid)), uintptr(unsafe.Pointer(&group)))
}

// Wait runs everything recorded and waits for it to finish. It returns the
// GPU's error, if any; the command buffer cannot be used again.
func (c *CommandBuffer) Wait() error {
	c.check("Wait")
	c.done = true
	defer c.pool.Pop()
	c.enc.Send(selEndEncoding)
	c.cb.Send(selCommit)
	c.cb.Send(selWaitUntilCompleted)
	if c.cb.Send(selStatus) == commandBufferStatusError {
		return fmt.Errorf("metal: %s", objc.ID(c.cb.Send(selError)).Description())
	}
	return nil
}
