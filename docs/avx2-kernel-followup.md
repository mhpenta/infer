# Follow-up: close the AVX2-only gap

Status: not started.

## Why

On CPUs with AVX-512, the 12×32 microkernel runs at ~92% of the measured
peak. On CPUs with only AVX2 and FMA, matrix products use the 6×16 AVX2
kernel, which runs at 69–76% of peak. BLIS reaches about 90% with the same
6×16 shape, so there is room. Cloud VMs are a mix of both kinds, so the AVX2
path matters.

On one AVX-512 CPU the AVX-512 peak (70.7 GFLOP/s) was about the AVX2 peak
(73.7): the 12×32 kernel's advantage there came from efficiency, not width.

## Ideas to try

1. **Prefetch** the next panel rows and strip values (`PREFETCHT0` a few k
   steps ahead) in `kernel6x16AVX2`.
2. **Different unrolling** of the k loop (2× or 8× instead of 4×), and
   interleaving the broadcasts with the FMAs differently, to keep both FMA
   ports busy.
3. **A 4×24 shape** (12 accumulators: 4 rows × 3 YMM columns). This changes
   the panel width (nr) for AVX2 machines, so weight packing would follow the
   chosen kernel.
4. **Edge rows on a wide path.** A 68-token input is 5 full 12-row tiles plus
   8 leftover rows, which go through 6×16 (7.5% of the time on one AVX-512
   CPU). A wide variant for the leftover rows (for example 8×32 or 6×32) would
   keep them on the efficient path.

## How to measure

- `ops.PeakGFLOPS`, `ops.KernelGFLOPS` and `ops.MatMulGFLOPS` measure the
  peak FMA rate, each microkernel and whole products on the machine they run
  on; the target is ~90% of peak for 6×16. On shared cloud vCPUs, interleave
  repeated measurements of the variants being compared and take medians:
  back-to-back runs drift with turbo and AVX-512 clock changes.
- Results only mean something on an AVX2-only CPU: check `ops.CPUFeatures()`.
- Correctness: `TestKernelMatchesPortable` on amd64 with
  `INFER_REQUIRE_AVX2=1` (an amd64 Docker container on Apple silicon exposes
  AVX2 through Rosetta).
