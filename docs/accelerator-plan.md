# GPU inference on Apple silicon: plan

Goal: run the Llama-family decoders (MiniCPM5, Qwen3) on the Apple silicon
GPU through Metal, from Go, with **no cgo**, checked against the same PyTorch
goldens as the CPU path. Along the way, set up how a model picks its
accelerator, so a second backend can slot in later.

Non-goals for now: a general tensor library or graph compiler, training,
other GPUs. The CPU path stays as it is and remains the reference.

Everything measured here is on an M2 Max (30-core GPU, 32 GB), macOS 26,
Go 1.26, Command Line Tools only (no Xcode).

## Why the GPU helps

The two phases of generation are limited by different things.

**Decoding is bound by memory bandwidth.** Each new token reads every weight
once and does about two flops per weight read. MiniCPM5-1B reads 1.76 GB of
bf16 weights per token (679M parameters in the layers, 200M in `lm_head`; the
embedding table is a lookup). Qwen3-1.7B reads 3.44 GB.

| | CPU today | GPU, spike's naive kernel | GPU ceiling at 300 GB/s |
|---|---|---|---|
| bandwidth from DRAM | ~96 GB/s (bf16 one-row kernel) | 297–311 GB/s | |
| MiniCPM5-1B decode | 22.2 ms/token (45 tokens/s) | | 5.9 ms/token (~170 tokens/s) |
| Qwen3-1.7B decode | 39.2 ms/token (25.5 tokens/s) | | 11.5 ms/token (~87 tokens/s) |

M2 Max's DRAM peaks around 400 GB/s. The CPU cores can't draw that much on
their own, but the GPU can get most of it. Expect 2.5–3.5× on decoding once
attention and the small kernels are counted.

**Prefill is bound by arithmetic.** A prompt's tokens share each weight read,
so the products become real matrix-matrix products. The NEON kernel does
about 50 GFLOPS per thread, roughly 400 on the performance cores. The GPU's
float32 peak is about 10.7 TFLOPS (30 cores × 128 lanes × 2 flops × 1.4 GHz).
Prefill is 67–105 tokens/s today; this is where the largest gain is.

**The memory is unified.** CPU and GPU share the same DRAM, so there is no
PCIe copy. A buffer the GPU reads can wrap memory Go already has: the spike
wraps both anonymous and file-backed `mmap` regions without copying. A
safetensors file that is already memory-mapped can be handed to the GPU as
it is.

## Calling Metal from Go without cgo

Metal is an Objective-C API. Every Objective-C method call is a call to one C
function, `objc_msgSend(receiver, selector, args...)`. If Go can call C
functions in system libraries, it can drive all of Metal through that one
function.

Go can do this without cgo. It is how `golang.org/x/sys/unix` calls
libSystem on macOS and how the standard library's `crypto/x509` calls
Security.framework. Three pieces:

1. **A dynamic import.** `//go:cgo_import_dynamic` is allowed in ordinary Go
   files, not only cgo-generated ones. It tells Go's internal linker to
   record a library and a symbol in the Mach-O binary, and dyld binds them
   when the program starts:

   ```go
   //go:cgo_import_dynamic objc_msgSend objc_msgSend "/usr/lib/libobjc.A.dylib"
   //go:cgo_import_dynamic MTLCreateSystemDefaultDevice MTLCreateSystemDefaultDevice "/System/Library/Frameworks/Metal.framework/Metal"
   ```

2. **An assembly trampoline** that jumps to the bound symbol, and a Go
   variable holding the trampoline's address (the same pattern as x/sys):

   ```asm
   TEXT objc_msgSend_trampoline<>(SB),NOSPLIT,$0-0
   	JMP	objc_msgSend(SB)
   GLOBL	·objc_msgSend_trampoline_addr(SB), RODATA, $8
   DATA	·objc_msgSend_trampoline_addr(SB)/8, $objc_msgSend_trampoline<>(SB)
   ```

3. **The runtime's libc call path**, `syscall.syscall9`, reached with
   `//go:linkname`. x/sys links to it, so the Go team keeps it available
   (it exists in Go 1.25 and 1.26). It switches to the system stack, puts up
   to nine integer arguments in the C calling convention, and marks the
   goroutine as blocked in a syscall, so a blocking call such as
   `waitUntilCompleted` does not stall the scheduler.

With those in place, `dev.send("newCommandQueue")` is an ordinary Go call.
The spike builds with `CGO_ENABLED=0`. `otool -L` shows the binary linking
libobjc, Metal and Foundation.

| Spike measurement | |
|---|---|
| one `objc_msgSend` from Go | 82–86 ns |
| compiling the embedded `.metal` source at run time | 55–540 ms (the OS caches compiled shaders) |
| bf16 matrix-vector product, 130,560 × 1,536 (400 MB, from DRAM) | 1.29–1.35 ms, 297–311 GB/s, max error 5e-7 against float64 |
| bf16 matrix-vector product, 9,216 × 1,536 (28 MB, fits the 48 MB system cache) | 97–133 µs |
| commit an empty command buffer and wait for it | 212–220 µs |
| `maxBufferLength`, `recommendedMaxWorkingSetSize` | 20.1 GB, 26.8 GB |

### What the bridge cannot do, and why it doesn't need to

- **No float arguments or float returns.** `syscall9` fills only the integer
  registers. Metal's compute API almost never takes a float argument.
  Scalars such as RMSNorm's ε, the attention scale and RoPE's θ go to
  kernels through `setBytes:length:atIndex:` as a small struct, which is
  what you'd do anyway. One float return worth having is `GPUStartTime` and
  `GPUEndTime` for profiling. A ten-line arm64 trampoline that returns `D0`
  can read them later. The project already writes arm64 assembly.
- **Structs over 16 bytes go by pointer.** arm64 passes them that way, so
  `MTLSize` (three 64-bit integers) is passed as a pointer to a copy.
- **No Objective-C blocks**, so no completion handlers. Use
  `waitUntilCompleted`, which blocks only the calling goroutine. If
  overlapping CPU work with the GPU is ever wanted, the alternative is
  `MTLSharedEvent` polling.
- **Memory rules.** Methods named `new…` return objects you own and must
  `release`. Others return autoreleased objects, which need an
  autorelease pool (`objc_autoreleasePoolPush`/`Pop`) around each forward
  pass. A pool belongs to an OS thread, so the goroutine that encodes holds
  `runtime.LockOSThread` for that span. A Go pointer passed as a `uintptr`
  could go stale if the goroutine's stack grows and moves during the call.
  `objc.Send` is marked `//go:uintptrescapes`, so the compiler moves
  anything converted with `uintptr(unsafe.Pointer(p))` in its argument
  list onto the heap and keeps it alive (step 0).
- **No SDK, no Xcode.** Nothing is compiled against Apple headers. Shaders
  are compiled at run time by the OS from source embedded with `go:embed`
  (`newLibraryWithSource:options:error:`). So the Metal language version is
  whatever the running OS supports, not what the Command Line Tools SDK
  has. The cgo version of the spike needed the SDK's headers and failed on
  a Metal API newer than the SDK; the cgo-free one did not.
- **CoreGraphics is not needed.** Older advice says command-line programs
  get no default device unless CoreGraphics is loaded. On macOS 26 the
  spike gets the device without it.

### Options considered

| Approach | Verdict |
|---|---|
| cgo with an Objective-C file | Ruled out by the project's no-cgo rule. It also ties builds to the SDK, and cross-compiling gets harder. |
| [purego](https://github.com/ebitengine/purego) (Ebitengine's library, which runs its Metal renderer this way) | The same mechanism, plus float arguments and callbacks. It works, but it is a dependency for about 150 lines this project can own. Worth keeping as a reference for edge cases. |
| **Our own bridge**: dynamic imports, trampolines, `syscall9` | **Chosen.** Small, has no dependencies, and Metal compute needs only integer and pointer arguments. The spike already uses it. |
| A Swift helper process, with commands over a pipe and tensors in shared memory | Works, and the Go binary is pure Go. But it means a second toolchain and a second binary to ship, a process to start, monitor and restart, and shared-memory setup for weights, KV caches and activations. It gains nothing the bridge doesn't already give. Not worth it unless the bridge hits a wall. |
| MetalPerformanceShaders / MPSGraph | Callable through the same bridge. Useful as a speed reference for the prefill product, not as the main path: it is less controllable and harder to check layer by layer. |

## The one rule: one command buffer per forward pass

A round trip to the GPU (commit, then wait) costs about 215 µs whatever the
work. A MiniCPM5-1B token is about 290 kernel dispatches: per layer,
RMSNorm, the qkv product, RoPE and cache append, attention, the output
product, an add, RMSNorm, gate+up, SwiGLU, down and an add; then the final
norm and `lm_head`. Waiting after each one would cost 62 ms per token,
against 22 ms for the whole token on the CPU today.

So a forward pass is encoded into one command buffer: every dispatch for
every layer, committed once and waited on once. The CPU reads back only the
logits row, which costs nothing on unified memory, or only the argmax when
decoding greedily. Encoding costs about 290 dispatches × ~5 messages × 85 ns
≈ 0.12 ms. If that ever matters, Metal's indirect command buffers can
replay a pre-encoded token.

This shapes everything below. Offloading single `ops.LinearInto` calls to
the GPU while the rest stays on the CPU would be slower than the CPU alone.
The model's whole forward pass moves at once.

## How a model picks its accelerator

```go
model, err := llama.LoadWith(dir, llama.Options{Device: accel.Auto}) // accel.Auto (default), accel.CPU, accel.Metal
fmt.Println(model.Device())                                           // "metal (Apple M2 Max)" or "cpu"
```

The choice is made in this order:

1. `Options.Device`, if set.
2. Otherwise the `INFER_DEVICE` environment variable, for tests and
   benchmarks.
3. Otherwise `auto`, which picks Metal when all of these hold: the build has
   it (darwin/arm64), a device opens, the startup self-check passes, and
   the model suits it. Suiting means bf16 weights, no `Int8`, shapes the
   kernels take, and weights within three quarters of
   `recommendedMaxWorkingSetSize`, leaving room for caches. If any of these
   fails, it picks the CPU.

Asking for `"metal"` explicitly and not getting it is an error, not a silent
fallback. The self-check runs each kernel on a small input against `ops` at
first use. This is the same idea as `ops.KernelCheck` gating the AVX-512
kernels, and it catches a driver or compiler surprise before it produces a
wrong answer.

The choice is made at load time because it decides how the weights are
prepared. The CPU path packs them into panels. The Metal path copies them
once, bf16 and row-major as stored, into GPU buffers. That is exactly the
layout the GPU kernels read, so a Metal model does no packing, and its
memory is the size of its weights. It reads them with
`safetensors.File.Bytes`, which returns the stored bytes without widening
them. Wrapping the mmap in place would save the copy, but q|k|v and gate|up
have to be stacked and offsets are only 8-byte aligned, so step 2 copies.

## Packages

| Package | Contents |
|---|---|
| `internal/objc` | The bridge: dynamic imports, trampolines, `Send`, autorelease pools. The real one is darwin/arm64 only; elsewhere `Available` is false and every call returns 0. |
| `metal` | A thin Go wrapper around the Metal objects: `Device`, `Buffer`, `Pipeline`, `Encoder`. It also embeds the kernels (`kernels/*.metal`) and provides `metal.Open()`. On other platforms `Open` returns `ErrUnavailable`, so every package still builds everywhere. |
| `accel` | The selection logic above. `Resolve(pref)` applies the option, then `INFER_DEVICE`, then `auto`. `OpenMetal()` opens the process's one GPU and compiles its kernels, running `Check` (every kernel against `ops`) on first use. Free of platform code, so a later backend adds one case. |
| `model/internal/gpu` | What the models' GPU paths share (step 5): a `Loader` that copies weights from safetensors into GPU buffers (linear layers in their stored dtype or int8, stacked, with biases; vectors, matrices and embedding tables), with a dry run that checks everything and counts bytes first; `Fits`; and `Scratch`, buffers for one call. |
| `model/llama`, `model/bert`, `model/t5` | Each has a second forward pass, `metal.go`, beside the CPU one. It records the same sequence of operations into one command buffer. In llama, the state it keeps (tokens, per-layer KV caches, `truncate`) has the same shape as the CPU's, so `Session` and prefix reuse work unchanged. |

**Don't build a general backend interface yet.** The CPU forward is about 40
lines. A Metal forward of the same shape is easier to read and to check than
an abstraction designed before a second model needs it. Once BERT or T5 move
to the GPU, the common operations will show what the interface should be.
(Step 5 did that; see what it extracted.)

## Data on the GPU

- **Weights.** One `MTLBuffer` wraps each safetensors file's mmap, and each
  tensor is a byte offset into it. Offsets are always 8-byte aligned (the
  header is padded to 8) but only sometimes 16-byte aligned: every tensor in
  MiniCPM5-1B is, none in MiniCPM5-2B or MiniLM are. So kernels load at
  most 8 bytes (four bf16 values) per lane per load, or the loader copies a
  misaligned file into fresh buffers. int8 weights are quantized at load
  into new buffers, as they are today.
- **KV cache.** Per layer, `[kvHeads, capacity, headDim]` float32 (fp16
  later, which halves it). `Truncate` is just a length change.
- **Activations.** float32 scratch buffers, allocated once per sequence and
  sized for the largest prefill chunk.
- **Logits.** The GPU writes the last position's row into a buffer the CPU
  reads in place, so `sample.Sampler` is unchanged.

## Kernels

GPU vocabulary: a *thread* runs one instance of a kernel. 32 threads form a
*simdgroup* that executes in lockstep and can reduce across itself in one
instruction (`simd_sum`). Up to 1,024 threads form a *threadgroup*, which
shares a fast on-chip memory. The grid is all the threadgroups of one
dispatch. The spike's matrix-vector product gives each simdgroup four output
rows. Its 32 lanes stride along k, reading eight bf16 weights per step and
widening them with a shift, exactly as the NEON kernel does. A `simd_sum`
then reduces each row.

Order of writing them, all in `metal/kernels/`:

| Kernel | For | Notes |
|---|---|---|
| `gemv_bf16` | decode products | The spike's kernel reaches 297–311 GB/s. Tuning goal 340+. It must handle row counts and k that aren't multiples of the tile. |
| `rmsnorm` | per layer, and Qwen3's per-head q/k norm | one threadgroup per row, sum of squares by `simd_sum` |
| `rope`, `kv_append` | RoPE on q and k; k and v written into the cache | two kernels for now (step 1); fuse them once dispatch count matters |
| `attend` | one query row against the cache (decoding), causal, with grouped-query attention | the softmax max and sum in one pass, online |
| `swiglu`, `add`, `embed` | elementwise, and the embedding lookup from the bf16 table | fuse the residual add into the following RMSNorm later |
| `argmax` | greedy decoding | reads back 4 bytes instead of 522 KB |
| `gemm_mma_bf16` | prefill products (step 3) | `simdgroup_matrix` 8×8 multiply-accumulate, tiles in threadgroup memory |
| `attend_tiled` | causal attention over a prompt (step 3) | tiled, FlashAttention-style, on the matrix units |
| `gemv`, `gemm_mma` for int8 | int8 weights (step 4) | the bf16 kernels templated on the weight type, one scale per output, as `ops.PackInt8` |

## Correctness

The GPU sums in a different order from the CPU kernels, so results won't be
bit-identical to the CPU path. They will be checked against PyTorch exactly
the way the CPU path is:

- **Each kernel against `ops`**, on random inputs that include awkward
  shapes (k and rows that aren't multiples of the tile, one row, long rows),
  with a tolerance relative to the value range. Fast math stays off.
- **The model against the goldens**: every layer's hidden states within a
  tolerance of their range, top-64 logits at every position, and generated
  tokens identical on the golden prompts (MiniCPM5-1B's four, up to 2,461
  tokens, and Qwen3's four). The CPU's tolerances (5.8e-6, 7.3e-6) are the
  bar to approach, not necessarily to meet.

**A decision to make: batch invariance.** On the CPU, a row computes
identically whatever rows run with it, so a session's reused prefix gives
bit-identical logits to running the whole prompt again. On the GPU, the
prefill product and the decode product will sum in different orders unless
built not to. There are two ways to go:

- (a) Accept matching within tolerance on the GPU.
- (b) Make every output element sum over k in a fixed blocking, whether it
  comes from the matrix-matrix or the matrix-vector kernel.

(b) is possible and keeps the property the CPU path has. It costs some
freedom in tuning the prefill kernel. Recommendation: build (a) first,
measure, then decide.

**Decided in step 2: (b), for now.** It came almost for free. The step 2
prefill product, `gemm_bf16`, sums each output exactly as `gemv_bf16` does,
and every other kernel works on one row at a time. So the GPU keeps the CPU
path's property, and `checkSession`'s bit-for-bit comparison passes on
Metal unchanged. Step 3's fast prefill product will sum in another order.
It must either keep (b), for example by having decode use the same kernel,
or the session check moves to a tolerance on the GPU.

**Decided in step 3: (a) on the GPU.** Keeping (b) would mean decoding with
the prompt kernels. At one row, `gemm_mma_bf16` is 4.8–6.6× slower than
`gemv_bf16` on every decoding shape (for example, `lm_head` takes 9.8 ms
against 2.0 ms), which would make decoding slower than the CPU. So on the
GPU a Session's logits match a fresh run's within float32 rounding:
- at most 4.3e-6 of the row's range over every golden case;
- exactly, when the reused prefix came from prompt passes alone.

`checkSession` holds the GPU to 1e-5 (`sessionTol`) and the CPU to bit for
bit. The CPU path's guarantee is unchanged. `Session`'s documentation says
which holds where.

**Testing on CI.** Tests that need a device skip without one, unless
`INFER_REQUIRE_METAL=1` makes them fail instead (as `INFER_REQUIRE_AVX2`
does for the x86 kernels). `macos-latest` runners are virtual machines whose
Metal support is limited. The first CI run with `metal` shows whether they
get a device (`TestOpen` logs it). If they don't, Metal correctness runs
locally before merging, like the 5 GB models that aren't in CI already.

## Steps

Each step ends with a test; nothing moves on until it passes.

### 0. Bridge and device (`internal/objc`, `metal`) — done

`internal/objc` has the dynamic imports, the trampolines, `ID.Send`,
`NewString`, and `PushPool`/`Pop`. `objc_other.go` covers every other
platform: `Available` is false and each call returns 0, so package
initialization runs anywhere and `metal.Open` returns `ErrUnavailable`.

`metal` is a thin wrapper, written once for all platforms:

| Type | |
|---|---|
| `Device` | `Open`, `Name`, `UnifiedMemory`, `MaxBufferLength`, `RecommendedMaxWorkingSetSize`, `Close` |
| `Library` | `Device.Compile(src)`, with fast math off; compile errors come back with the compiler's message |
| `Pipeline` | `Library.Pipeline(name)`, `MaxThreadsPerThreadgroup`, `ThreadExecutionWidth` |
| `Buffer` | `Device.NewBuffer(n)`, shared memory; `Device.Wrap(b)`, zero-copy over page-aligned mmap'd memory; `Bytes`, `metal.Slice[T]` |
| `CommandBuffer` | `Device.NewCommandBuffer`, `SetPipeline`, `SetBuffer(index, b, offset)`, `SetBytes`, `metal.SetValue` for parameter structs, `DispatchThreads`, `DispatchThreadgroups`, `Wait` |

A `CommandBuffer` holds one serial compute encoder, so each dispatch sees
the writes of the ones before it. From `NewCommandBuffer` to `Wait`, its
goroutine is locked to its thread, which holds the autorelease pool for the
command buffer's Metal objects. Misuse panics before Metal sees it, as `ops`
panics on shape mismatches: a buffer index or offset out of range, a
dispatch with no pipeline or with too many threads per group, a call after
`Wait`. With no cgo, an Objective-C exception from Metal could not be caught
and would abort the process.

Tests, passing with `CGO_ENABLED=0` on the M2 Max:

- Vector add on 1,000,003 elements, bit-identical to Go's float32 add.
- A second dispatch in the same command buffer, reading the first one's
  output through buffer offsets.
- A parameter struct passed by value.
- The GPU writing into a wrapped anonymous mapping, plus rejection of
  misaligned memory.
- Compile errors and missing kernels.
- Eight goroutines running command buffers at once.
- NSString round trips with non-ASCII text, and a seven-argument message.

A kernel altered to be off by one ulp fails the add test. The race detector
is clean (it needs cgo, so that run is the only one with cgo on).

Checked on Go 1.25.5, which CI uses, and Go 1.26. `go vet ./...` is clean on
darwin/arm64, linux/amd64, linux/arm64, darwin/amd64 and windows/amd64. In
the linux/amd64 Docker image, every Metal test skips with "no Metal device".

Findings:

- **`//go:nosplit` was the wrong protection for pointers.** The first
  version of `call` was nosplit, so the stack could not move between the
  pointer conversion and the C call. On Go 1.26 that links. On Go 1.25 the
  linker rejects it: the runtime chain under `syscall9` goes 8 bytes over
  the nosplit stack limit. `//go:uintptrescapes` is allowed outside the
  standard library and does not depend on the runtime's stack use. Escape
  analysis (`go build -gcflags=-m`) shows the `NSError` out-parameters and
  the `MTLSize` arguments moved to the heap, as intended.
- **No leaks.** Peak memory is 19.2 MB after 2,000 command buffers and
  19.5 MB after 40,000, so the autorelease pools drain.
- **The fixed cost** of an empty command buffer (`BenchmarkRoundTrip`) is
  175–199 µs, as the spike measured.
- **Compiling a library** takes 55–540 ms. The OS caches compiled shaders,
  so only a cold run is slow. A model compiles once at load.

### 1. Decode kernels (`metal`) — done

The kernels are in `metal/kernels/` (`common.metal`, `gemv.metal`,
`elementwise.metal`, `attention.metal`). They are compiled together by
`Device.Kernels()`, and each has a Go method that checks shapes, bounds and
alignment, then records one dispatch. Operands are `metal.Ref`s, a buffer
and a byte offset, so a kernel can read the key part of a stacked q|k|v row
in place. `metal.KVCache` is one layer's cache: two `[heads, capacity, dim]`
buffers. `ops.Rotation.Tables()` is new: it hands the CPU's cosine and sine
tables to the GPU.

Each kernel against `ops`, on the M2 Max:

| Kernel | Matches `ops` | Largest error seen | How it works |
|---|---|---|---|
| `GemvBF16` | within float32 rounding of a float64 reference | 1.0–1.7e-8 of Σ\|w·x\| on the model shapes (NEON: 3.7–6.0e-8) | each simdgroup owns 4 rows; each lane reads 8 bf16 weights (two 8-byte loads) per step, so a simdgroup streams 512 contiguous bytes of a row; a tail loop handles k not a multiple of 256 (k must be a multiple of 4); rows past n reread the last row instead of branching |
| `RMSNorm` | within 1e-6 relative | 1.1e-7 of the largest value | one threadgroup per (row, group); float32 tree sum, against `ops`' float64 sum; also does Qwen3's per-head norms in place on stacked q\|k\|v rows |
| `RoPE` | **bit-identical** | 0 | reads `ops.Rotation`'s tables, and writes each product as `fma(x, c, 0)` so the compiler cannot fuse it into the sum; query and key heads rotate in one dispatch |
| `KVAppend` | **bit-identical** | 0 | a copy |
| `Attend` | within 2e-6 | 1.0e-6 of the largest value, after 2,500 positions | one threadgroup per (head, row) of 8 simdgroups; each simdgroup runs an online softmax over every 8th position (lane l holds dimensions l, l+32, …), then the 8 combine |
| `SwiGLU` | within 1e-6 relative | 2.9e-8 | Metal's `exp` against `ops`' `expf` |
| `Add`, `EmbedBF16`, `Argmax` | **bit-identical** | 0 | `Argmax` runs in one threadgroup and breaks ties to the first index, as `ops.Argmax` does |

The GEMV cases include MiniCPM5-1B's shapes, shapes that aren't multiples of
the tiles (37×100, 5×260, 131×1,028, 1×4), and weights at an offset that is
8- but not 16-byte aligned, as in MiniCPM5-2B's files. The attention cases
are:
- a first token;
- decoding after 300 and 2,500 positions;
- seven causal rows after 40;
- 16 query heads over 2 key/value heads (MiniCPM5);
- no grouping;
- heads of 32 and 64 values.

Two mutations confirm the tests catch errors: dropping the GEMV tail loop
fails it, and letting attention see one position past the mask fails it.

**GEMV speed.** Measured with Low Power Mode **on** (`pmset -g` shows
`lowpowermode 1`), which caps the GPU under sustained load. Every run cycles
through 512 MB of distinct weight copies, so both sides read from DRAM:

| MiniCPM5-1B product | GPU | NEON, all cores | |
|---|---|---|---|
| q\|k\|v, 2,560 × 1,536 | 46 µs, 171 GB/s | 199 µs, 40 GB/s | 4.3× |
| output, 1,536 × 2,048 | 45 µs, 140 GB/s | 156 µs, 40 GB/s | 3.5× |
| gate\|up, 9,216 × 1,536 | 195 µs, 146 GB/s | 642 µs, 44 GB/s | 3.3× |
| down, 1,536 × 4,608 | 106 µs, 134 GB/s | 516 µs, 27 GB/s | 4.9× |
| `lm_head`, 130,560 × 1,536 | 2.77 ms, 145 GB/s | 10.5 ms, 38 GB/s | 3.8× |

Under this cap, one token's products would take 12.2 ms on the GPU against
47 ms on NEON. With power unconstrained the spike's kernel reached 297–311
GB/s, against the 96 GB/s the README gives for NEON. The ratio should hold
near 3×; the absolute numbers will roughly double.

Low Power Mode also changes results with run length. The same product ran
at 61 GB/s when timed over a single command buffer, 203 GB/s over 20 (about
0.4 s), and 155 GB/s sustained for 2 s. So `gemv.metal`
instantiates five shapes (2, 4 or 8 rows per simdgroup, 4 or 8 simdgroups
per threadgroup), and under the cap they all land within noise of each other
(134–154 GB/s). The kernel keeps 4 rows × 8 simdgroups, the spike's shape,
until `BenchmarkGemvVariants` runs on mains power with Low Power Mode off.

```bash
go test -run XXX -bench 'GemvBF16|GemvVariants' ./metal
```

Left for later, with the forward pass to measure against:
- fusing RoPE with the cache append, and the residual add with RMSNorm;
- attention that reads a key/value head once for its whole query group
  (today each of the 8 query heads in a group reads it, mostly from cache);
- splitting long contexts across threadgroups.

### 2. MiniCPM5-1B on the GPU (`model/llama`, `accel`) — done

**The forward pass** (`model/llama/metal.go`). `openGPU` first does a dry
run over every tensor, checking dtype and shape and adding up bytes against
the working set, before anything is allocated. Then it copies the weights
into GPU buffers. A forward pass is one command buffer per chunk of up to
512 positions:
1. the embedding lookup;
2. for each layer:
   - RMSNorm, the q|k|v product, Qwen3's per-head norms, and RoPE on query
     and key heads together;
   - the cache append, attention, the output product, and the residual add;
   - RMSNorm, gate|up, SwiGLU, down, and the add;
3. the final norm;
4. when generating, `lm_head` on the last position only.

A decoding step is one round trip, and the CPU reads 522 KB of logits in
place from shared memory. `state.next` is the step both paths share, so
`generate`, `Session` and the sampler are unchanged.

GPU memory is released by `Model.Close`, by a new `Session.Close`, and
after each `Generate`. `runtime.AddCleanup` releases it as a backstop when a
state is collected without being closed.

**The prefill product** is `gemm_bf16`: `gemv_bf16` applied to blocks of 8
rows, each weight loaded once per block. Each row of its output is
bit-identical to `gemv_bf16` on that row (`TestGemmBF16`), which keeps
Session reuse bit-identical (see batch invariance above; step 3 gave that
up for speed).

**Selection** (`accel`): `Options.Device`, then `INFER_DEVICE`, then
`auto`. The first `OpenMetal` compiles the kernels and runs `accel.Check`,
which compares every kernel against `ops` and takes about 150 ms. A RoPE
kernel written as `x1*c - x2*s` fails it: the compiler fuses the product
into the sum, which is why the kernel writes `fma(x, c, 0)`.

`TestDevice` covers:
- an unknown device is rejected;
- `INFER_DEVICE=cpu` is honored;
- the default picks Metal when it is available;
- asking for Metal with int8 weights is an error.

The int8 tests load with `auto`, so they cover the fallback to the CPU.

**The goldens on Metal.** `TestMatchesPython` now runs every check once per
device: CPU, then Metal when present (`INFER_REQUIRE_METAL=1` makes its
absence a failure). Both models pass every check on the GPU, including the
2,461- and 2,579-token prompts. So Qwen3's per-head norms and tied
embeddings, planned for step 4, already work.

| Largest difference from PyTorch, of a tensor's or row's range | MiniCPM5-1B, Metal | Qwen3-1.7B, Metal |
|---|---|---|
| hidden states, every layer | 5.6e-6 (CPU: 5.8e-6) | 7.2e-6 (CPU: 7.3e-6) |
| top-64 logits after the first position | 1.0e-5 | 2.1e-5, at the first position (the attention sink, allowed 3e-5); every later position within 2e-5 |
| generation steps | 4.3e-6 | 1.0e-5 |

On Metal, generated tokens match PyTorch's on every prompt, and every
Session step is bit-identical to a run from an empty cache.

**Speed**, measured on battery with Low Power Mode **on**. That mode
throttles the CPU much harder than the GPU: the CPU decodes MiniCPM5-1B at
90 ms/token here against 22 ms normally. So the ratios below overstate the
gain. Rerun plugged in, with Low Power Mode off:

```bash
go test -run XXX -bench 'Decode|Prefill' ./model/llama
```

| | CPU | Metal |
|---|---|---|
| MiniCPM5-1B decode | 89.9 ms/token | 13.5 ms/token (74 tokens/s) |
| Qwen3-1.7B decode | 186 ms/token | 28.9 ms/token (35 tokens/s) |
| MiniCPM5-1B prefill, 65 tokens | 1,498 ms (43 tokens/s) | 330 ms (196 tokens/s) once warm; 1,175 ms on a cold first call |
| MiniCPM5-1B prefill, 2,461 tokens | 59.0 s (42 tokens/s) | 15.7 s (157 tokens/s) |

On the step 1 GEMV numbers (also throttled), one token's products would take
12.2 ms. Decoding at 13.5 ms means the round trip, the small kernels and
attention add only about 1.3 ms. The 2.5× target can't be judged
until the CPU runs unthrottled. On the spike's unthrottled bandwidth the
products alone would take about 6 ms (165 tokens/s).

Prefill is the weak spot, as expected of the naive product. Its 8-row
blocks reread x for every 4 weight rows, and it runs at roughly 100 GFLOPS
where the GPU can do thousands. Step 3 replaces it.

### 3. Fast prefill (`metal`) — done

**The product**, `gemm_mma_bf16` (`kernels/gemm.metal`), is used for every
prompt product (`Kernels.GemmMMA`):
- Each threadgroup takes a 64×64 block of the output and steps through k 16
  at a time.
- Its 256 threads copy the 64×16 blocks of x and W (W widened from bf16)
  into threadgroup memory, zero-filling past the edges.
- Each of its 2×4 simdgroups accumulates a 32×16 share from 8×8
  `simdgroup_multiply_accumulate`s, the W tiles loaded transposed.

`TestGemmMMA` checks it against float64 on prompt shapes and on edges that
cut through every tile. Its error is up to 3e-7 of Σ|w·x|, about 10× the
GEMV's: each output is one sequential chain along k instead of a 32-way
tree. That is still well inside the bound, and 70× below the tolerance
against PyTorch.

Tile shape decided the speed. `BenchmarkGemmVariants`, under Low Power
Mode:

| MiniCPM5-1B product | step 2 `gemm_bf16` | 32×16 per simdgroup (chosen) | 32×32 per simdgroup |
|---|---|---|---|
| gate\|up, 65 rows | 230 GFLOPS | 1,446 GFLOPS | 290 GFLOPS |
| down, 65 rows | 237 | 1,246 | 224 |
| gate\|up, 512 rows | 254 | 2,375 | 573 |
| down, 512 rows | 342 | 2,074 | 494 |

A 32×32 share needs 16 accumulator tiles, which don't fit in registers. The
same shape measured about half as fast in one earlier run, because Low
Power Mode's throttling varies run to run. Compare within a run only.

**Attention.** With the fast product in place, a temporary switch that
skipped attention showed attention was **81%** of the 2,461-token prompt's
7.0 s (and 16% of the 65-token one's). The step 1 kernel gives each query
row a threadgroup that reads the whole cache. `attend_tiled` replaces it
for prompts:
- A threadgroup takes 32 query rows of one head.
- Each of its 4 simdgroups keeps 8 rows' queries and running context as
  8×8 matrices.
- The threadgroup streams the cache through threadgroup memory 16
  positions at a time.
- Per step, a simdgroup multiplies out its scores. Four lanes per row mask,
  scale and fold them into the row's running maximum and sum, writing the
  weights back.
- The context is rescaled by multiplying with a diagonal matrix of the
  corrections, then the weights' product with the values is added.

`Kernels.Attend` picks it for more than one row with heads of up to 128
values; decoding keeps `attend`. `TestAttend` now includes a 300-row prompt,
100 rows after a 37-position prefix (blocks that don't line up), a 512-row
chunk after 1,500 positions (within 1.7e-6 of `ops`), and 64-value heads.
Letting rows see one position past the mask fails it.

**Results.** Both models pass every golden on Metal, including both long
prompts. Hidden states are within 5.1e-6 (MiniCPM5-1B) and 3.8e-6
(Qwen3-1.7B) of PyTorch. Qwen3's first-position attention sink, at 2.1e-5 in
step 2, is now 6.3e-6. The Metal half of the golden tests runs in 72 s
instead of 195 s. Prefill, under Low Power Mode:

| | step 2 | step 3 | CPU (throttled, step 2) |
|---|---|---|---|
| MiniCPM5-1B, 65 tokens | 330 ms (196 tokens/s) | 61–96 ms (670–1,050 tokens/s) | 1,498 ms |
| MiniCPM5-1B, 2,461 tokens | 15.7 s (157 tokens/s) | 2.26 s (1,091 tokens/s) | 59.0 s |
| Qwen3-1.7B, 66 tokens | | 323 ms | |
| Qwen3-1.7B, 2,579 tokens | | 5.4 s (478 tokens/s) | |

Decoding is unchanged by this step.

**Both devices on AC power**, with nothing else running (Low Power Mode
still on, but on AC it barely slows the CPU: MiniCPM5-1B decodes at 24.6
ms/token against the usual 22). This is the fairest comparison so far,
`go test -run XXX -bench 'Decode|Prefill' ./model/llama`:

| | CPU | Metal | |
|---|---|---|---|
| MiniCPM5-1B decode | 24.6 ms/token (41 tokens/s) | 9.0 ms/token (112 tokens/s) | 2.7× |
| Qwen3-1.7B decode | 76.3 ms/token | 23.5 ms/token (43 tokens/s) | 3.2× |
| MiniCPM5-1B prefill, 65 tokens | 996 ms | 149 ms (434 tokens/s) | 6.7× |
| MiniCPM5-1B prefill, 2,461 tokens | 31.4 s (78 tokens/s) | 2.38 s (1,032 tokens/s) | 13× |
| Qwen3-1.7B prefill, 66 tokens | 1,836 ms | 333 ms (198 tokens/s) | 5.5× |
| Qwen3-1.7B prefill, 2,579 tokens | 49.1 s (53 tokens/s) | 3.32 s (776 tokens/s) | 15× |

MiniCPM5-1B decoding meets step 2's 2.5× target. It streams 1.76 GB per
token in 9.0 ms, 196 GB/s. One run that overlapped the test suite still
decoded at 7.8 ms/token (128 tokens/s), close to the roughly 6 ms the
products take at the spike's 300 GB/s. Qwen3-1.7B streams 3.44 GB in 23.5
ms, only 146 GB/s, so something besides its products costs time there,
probably attention over 8 key/value heads; it's worth profiling. Benchmarks
run alongside the test suite are meaningless: one such run had the CPU
decoding Qwen3 at 179 ms/token and Metal prefilling MiniCPM5-1B's long
prompt in 12 s.

Not done from this step's plan: measuring MPS's matrix product as a
ceiling, which would need MPS through the bridge. What's left in
long-prompt prefill:
- About 1.34 s of products. They reach 2.1–2.4 TFLOPS here, against a
  float32 peak of about 10.7; double-buffering the tile loads is the usual
  next step.
- About 0.9 s of attention. Sharing a key/value head's tiles across its
  query group is the obvious next step there.

### 4. int8 weights and MiniCPM5-2B on the GPU — done

**Kernels.** `gemv` and `gemm_mma` are now templates on the weight type.
`widen4` loads four weights and widens them: bf16 by a shift, int8 by
converting its integer. int8 outputs are multiplied by their row's scale
once the sum is done, which is `ops.PackInt8`'s definition. New launchers:
`Kernels.GemvInt8` and `Kernels.GemmInt8`.

Renames along the way:
- The step 3 matrix-unit product is now `Kernels.GemmBF16`.
- The step 2 `gemm_bf16`, kept only because it summed like the GEMV,
  is removed.
- `LogitsInto` on the GPU runs the one-row kernel on each row, so it still
  computes exactly what generation computes, for bf16 and int8.

`TestGemvInt8` and `TestGemmInt8` check both kernels against float64 with
the scaling's rounding added to the bound. The int8 GEMV is again about 3×
closer to the exact result than NEON's (1.2–3.0e-8 of scale·Σ\|q·x\|,
against 2.6e-8–1.8e-7). Dropping the scale fails both tests, and
`accel.Check` now checks all four products against float64.

**Loading.** `ops.QuantizeInt8` is new: `PackInt8`'s quantization, row-major,
with the same integers and scales (`TestQuantizeInt8`). With
`Options.Int8`, the GPU loader quantizes each linear tensor, and stacks
q|k|v and gate|up after quantizing; per-row scales make that the same as
quantizing the stack. Qwen3's tied output layer gets its own int8 copy
while the embedding stays bf16, as on the CPU. int8 takes linear weights in
any dtype, but the embeddings must still be bf16.

**Selection.** `Auto` with `Int8` now runs on Metal. Nothing in the real
models falls back to the CPU any more, so `TestDeviceFallback` builds a
one-layer model with float32 weights in a temporary directory. Auto must
load it on the CPU, with or without int8. Asking for Metal must fail with a
message about bf16.

**Accuracy.** `TestInt8MatchesBF16` now runs per device, comparing int8
against bf16 on the same device. On Metal it reproduces the CPU's figures in
`docs/int8-weights.md` to the digit: 94.1–100% top-1 agreement, mean KL
0.0003–0.0072 nats, the same perplexities, and PyTorch's greedy tokens on
all four cases, the 2,461-token one included.

**MiniCPM5-2B.** It has no logit goldens (its check against PyTorch was
manual), so `TestMiniCPM5For2BOnMetal` compares the GPU with the CPU on
MiniCPM5-1B's golden prompts (same tokenizer). Greedy tokens are identical
on all three, and every step's logits are within 6.5e-6 of the CPU's,
against a tolerance of 2e-5. It skips with `-short` or without Metal; at
5 GB per device it is not for CI.

**Speed.** Measured with Low Power Mode off, on AC power, but on a busy
machine: load average 16–19, with 27 GB of swap in use by other processes.
The GPU figures agree with an idle run (MiniCPM5-1B bf16 at 8.4 ms/token,
against 9.0 idle). The CPU figures from that run are not usable: the 5 GB
bf16 2B model paged, at 472 ms/token. So the CPU column is the
quiet-machine figures from `docs/int8-weights.md`.

`go test -run XXX -bench DecodeWeights ./model/llama`:

| Decode | CPU bf16 | CPU int8 | Metal bf16 | Metal int8 | int8 on Metal |
|---|---|---|---|---|---|
| MiniCPM5-1B | 22.2 ms/token | 20.6 ms/token | 8.4 ms/token (119 tokens/s) | 6.1 ms/token (163 tokens/s) | 1.37× |
| MiniCPM5-2B | 65 ms/token (15.3 tokens/s) | 57 ms/token (17.5 tokens/s) | 21.6 ms/token (46 tokens/s) | 13.1 ms/token (76 tokens/s) | 1.65× |

On NEON, int8 was limited by instructions and decoded only 1.08–1.14×
faster than bf16. On the GPU, widening is cheap next to the memory traffic,
so int8 gets most of the gain halving the bytes promises. The rates are
1.76 GB of bf16 per token at 210 GB/s for MiniCPM5-1B, and 4.5 GB at 208 GB/s
for MiniCPM5-2B; int8 reaches 145–168 GB/s. MiniCPM5-2B int8 on the GPU
decodes 4.4× faster than on NEON.

### 5. BERT and T5 on the GPU, and what the three models share — done

**What the models share** turned out to be two things, both now extracted:
- **The kernels' interface** (`metal`):
  - One `Kernels.Linear(y, ldy, x, ldx, rows, Weights)` replaces the four
    product launchers. `Weights` names the buffer, the type (bf16, fp16,
    float32 or int8, with int8's scale) and an optional bias, and picks the
    one-row or many-row kernel.
  - One `Kernels.Attend(..., AttendArgs)` covers causal decoders, encoders
    that see every position, and cross-attention over another sequence's
    keys, with an optional additive bias.
- **Weight loading** (`model/internal/gpu`). The loader that was llama's,
  generalized to dtypes, biases and name prefixes. `llama` now uses it too,
  so its GPU path takes fp16 and float32 checkpoints as well as bf16.

The forward passes stay per model, each about the length of its CPU twin.

**New kernels**, each tested against `ops`:

| Kernel | For | Against `ops` |
|---|---|---|
| `gemv`, `gemm_mma` for fp16 and float32, all with a bias epilogue | BERT (MiniLM float32, mxbai fp16), T5 (float32) | within float32 rounding of float64 |
| `layernorm`, with an optional fused residual add | BERT's post-LN blocks | 1.5e-7 of the largest value |
| `gelu` | BERT's feed-forward | **bit-identical**: the same erf polynomial as `ops` |
| `relu`, `add_row`, `scale`, `embed` for fp16 and float32 | T5's feed-forward, BERT's token types, T5's tied output layer, both models' tables | **bit-identical** |
| `attend`, `attend_tiled` with `causal` off, `keys` and `bias` | encoders, cross-attention, T5's relative position bias | 1.4e-7 to 8.4e-7 of the largest value |

Leaving out the bias in the tiled kernel fails `TestAttendModes`. The
self-check (`accel.Check`) covers the reworked API.

**BERT** (`model/bert/metal.go`). One command buffer per forward pass:
1. The embedding lookup, plus the position rows, plus the token-type row,
   then LayerNorm.
2. Per layer:
   - the q|k|v product with its biases;
   - keys and values into a scratch cache, then attention over every
     position;
   - the output product, then LayerNorm with the residual fused in;
   - the intermediate product, GELU, the output product, then LayerNorm
     with the residual.

Every layer writes a new slot, so `HiddenStates` needs no extra round
trips. Workspaces are kept on a free list and released by `Close`.

On Metal, MiniLM passes every layer's check (atol and rtol 1e-4), with
pooled cosine 1.000000000. mxbai's binary embeddings match PyTorch **bit for
bit on all 20 texts**, as on the CPU.

**T5** (`model/t5/metal.go`):
- **The encoder and the teacher-forced decoder** are one command buffer
  each. Each layer writes a new slot, its residual adds reading the
  previous one, so the hidden states come free. The relative position bias
  tables stay on the CPU, which builds each call's `[heads, queries, keys]`
  bias as before; the GPU adds it to the scaled scores.
- **Cached generation** computes every decoder layer's cross-attention keys
  and values from the encoder output once, then runs one command buffer per
  token, through the tied output layer (scaled by d_model^-0.5 into a
  scratch row, so the hidden state stays unscaled).
- **Growth:** the self-attention caches and the bias buffer grow when full.
  `TestCachedStepsMatchTeacherForcing` starts them at one position.

`newDecoderState` and `step` now return errors (only the GPU fails).

On Metal, every encoder and decoder layer and the logits are within
1.7e-6 of PyTorch (relTol 2e-5). Cached and uncached greedy generation give
PyTorch's tokens on all 5 inputs, up to 512 tokens.

**Tests** now run once per device: `TestMiniLMMatchesPython`,
`TestMxbaiMatchesPython`, `TestMxbaiBinaryMatchesPython`,
`TestT5SmallMatchesPython`, both generation tests and the cached-steps test.

**Speed: indicative only.** The machine was under heavy outside load during
the run (load average 25–35, 22.8 of 23.5 GB swap in use), so neither device
ran at its normal speed. The CPU's T5 times here are almost twice the
README's (75 against 42 ms for the 6-token case), and a 3-token MiniLM pass
took the GPU 10 ms where one round trip costs 0.2 ms.

| | CPU | Metal |
|---|---|---|
| MiniLM, 512 tokens | 390 ms | 25 ms |
| mxbai, 65 tokens | 1,536 ms | 82 ms |
| mxbai, 29 tokens | 559 ms | 59 ms |
| t5-small generate, 512 in / 41 out | 1,371 ms | 383 ms |
| t5-small generate, 50 in / 62 out | 762 ms | 310 ms |
| t5-small generate, 11 in / 6 out | 75 ms | 65 ms |

The direction is clear. Long inputs and the larger model gain the most. For
t5-small's short cases the round trip per token dominates, since the model
is tiny. Rerun on a quiet machine:

```bash
go test -run XXX -bench 'MiniLMForward|MxbaiForward' ./model/bert
go test -run XXX -bench Generate ./model/t5
```

### Later: other accelerators

The selection layer is designed so a second backend adds one case. Expect it
to be harder elsewhere. On Linux, Go doesn't link libc without cgo, so
calling `libcuda` or Vulkan needs what purego calls "fakecgo". That is a
separate project, to start once Metal has shown what the backend interface
needs.

## The spike

[`metal-spike/`](metal-spike) is a separate module, like `benchcmp/`, so
`go test ./...` never builds it. It has the bridge (`objc_darwin.go`,
`objc_darwin_arm64.s`), the kernels (`kernels.metal`) and the measurements
above (`main.go`). It is throwaway code: no releases, fixed shapes, and the
selector cache is not safe for concurrent use. Run it with:

```bash
cd docs/metal-spike && CGO_ENABLED=0 go run .
```
