package ops

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/tensor"
)

// The matrix product follows the design of BLIS and OpenBLAS:
//
//   - The right-hand matrix is packed into panels of nr columns, each stored
//     k-major, so the microkernel streams it contiguously. Weights never
//     change, so they are packed once at load time (Pack).
//   - The left-hand matrix is packed once per product into strips of mr rows,
//     also k-major, so the microkernel reads a as a single stream too.
//   - The microkernel computes an mr×nr block of the output as a sum of outer
//     products: per k step it loads one panel row (nr values) and broadcasts
//     mr values of a, keeping all mr×nr sums in registers. Its epilogue writes
//     the block straight into c (store, store plus bias, or accumulate).
//   - With AVX-512, full 12×32 tiles (two strips by two panels) go to a wider
//     kernel with 24 accumulators; edge tiles keep the 6×16 kernel.
//   - Loops are blocked so a kc-deep slice of one panel (16 KiB) stays in L1
//     while it is reused for every mr-row strip, and the packed mc×kc block of
//     a stays in L2 while it is reused for every panel of an nc-wide block.
const (
	mr = 6  // output rows per microkernel call
	nr = 16 // output columns per microkernel call (packed panel width)
	kc = 256
	mc = 72  // a multiple of 2*mr, for the wide kernel
	nc = 128 // a multiple of 2*nr, for the wide kernel

	// Microkernel epilogues: what it does with its mr×nr sums.
	storeC     = 0 // c = sums
	storeBiasC = 1 // c = sums + bias (bias holds nr values)
	addC       = 2 // c += sums

	// macsPerWorker is the work (multiply-adds) that justifies waking one
	// more thread. Waking a parked thread costs microseconds, which dominates
	// the many small products of a short sequence.
	macsPerWorker = 1 << 19
)

// Packed is the right-hand operand of a product, n output columns over a
// shared dimension k, laid out for the microkernel: panel p holds columns
// [p*nr, p*nr+nr) with element (kk, jj) at data[p*k*nr + kk*nr + jj]. The
// last panel is zero-padded. Weights stored as fp16 can keep their panels in
// fp16 (half, same layout; see PackHalf), halving what a product streams.
type Packed struct {
	n, k int
	data []float32 // float32 panels, nil when half is used
	half []uint16  // fp16 panels
}

// N returns the number of output columns.
func (p *Packed) N() int { return p.n }

// K returns the shared (inner) dimension.
func (p *Packed) K() int { return p.k }

// Pack packs an nn.Linear weight w [out, in] so that MatMulPacked(x, Pack(w))
// computes x·wᵀ.
func Pack(w *tensor.Tensor) *Packed {
	return packT(w.Data, w.Rows(), w.Cols(), w.Cols())
}

// PackHalf is Pack with the panels stored as fp16. Every value of w must be
// exactly representable in fp16, as weights loaded from an F16 file are; the
// kernels widen each value back to the same float32, so products are
// bit-identical to Pack's while reading half the memory. It fails on a value
// that would round.
func PackHalf(w *tensor.Tensor) (*Packed, error) {
	n, k := w.Rows(), w.Cols()
	p := &Packed{n: n, k: k, half: make([]uint16, packedSize(n, k))}
	for j := range n {
		base := (j/nr)*k*nr + j%nr
		for kk, v := range w.Row(j) {
			h, ok := half.FromFloat32Exact(v)
			if !ok {
				return nil, fmt.Errorf("ops.PackHalf: w[%d][%d] = %g is not exact in fp16", j, kk, v)
			}
			p.half[base+kk*nr] = h
		}
	}
	return p, nil
}

// Half reports whether the panels are stored as fp16.
func (p *Packed) Half() bool { return p.half != nil }

// packT packs b holding n rows of k values (row j at b[j*ldb]), for a·bᵀ.
func packT(b []float32, n, k, ldb int) *Packed {
	p := packTInto(make([]float32, packedSize(n, k)), b, n, k, ldb)
	return &p
}

// packTInto is packT into dst, which must hold packedSize(n, k) values and
// may contain anything.
func packTInto(dst, b []float32, n, k, ldb int) Packed {
	p := newPackedIn(dst, n, k)
	for j := range n {
		base := (j/nr)*k*nr + j%nr
		for kk, v := range b[j*ldb : j*ldb+k] {
			p.data[base+kk*nr] = v
		}
	}
	return p
}

// packNInto packs b holding k rows of n values (row kk at b[kk*ldb]), for
// a·b, into dst, which must hold packedSize(n, k) values and may contain
// anything.
func packNInto(dst, b []float32, k, n, ldb int) Packed {
	p := newPackedIn(dst, n, k)
	for kk := range k {
		for j, v := range b[kk*ldb : kk*ldb+n] {
			p.data[(j/nr)*k*nr+kk*nr+j%nr] = v
		}
	}
	return p
}

// packedSize is the number of values a packed n×k operand occupies.
func packedSize(n, k int) int {
	return (n + nr - 1) / nr * k * nr
}

// newPackedIn wraps dst as a packed n×k operand, zeroing the padding columns
// of the last panel; packers then write every other value.
func newPackedIn(dst []float32, n, k int) Packed {
	data := dst[:packedSize(n, k)]
	if n%nr != 0 {
		clear(data[(n/nr)*k*nr:])
	}
	return Packed{n: n, k: k, data: data}
}

// MatMulPacked returns a·B for a [m, k] and packed B, as a new [m, n] tensor.
func MatMulPacked(a *tensor.Tensor, b *Packed) *tensor.Tensor {
	c := tensor.New(a.Rows(), b.n)
	MatMulPackedInto(c, a, b)
	return c
}

// MatMulPackedInto sets c [m, n] to a·B for a [m, k] and packed B.
func MatMulPackedInto(c, a *tensor.Tensor, b *Packed) {
	m := a.Rows()
	if a.Cols() != b.k || c.Rows() != m || c.Cols() != b.n {
		panic(fmt.Sprintf("ops.MatMulPackedInto: c is %v, a is %v, b is %dx%d", c.Shape, a.Shape, b.n, b.k))
	}
	gemm(m, a.Data, b.k, b, nil, c.Data, b.n, workersFor(m*b.n*b.k))
}

// scratch recycles temporary float32 buffers, such as attention's per-head
// scores and packed operands, across calls.
var scratch = sync.Pool{New: func() any { return new([]float32) }}

// getScratch returns a pooled buffer of length n with arbitrary contents.
// Return it with scratch.Put when done.
func getScratch(n int) *[]float32 {
	buf := scratch.Get().(*[]float32)
	if cap(*buf) < n {
		*buf = make([]float32, n)
	}
	*buf = (*buf)[:n]
	return buf
}

// MatMulT returns a·bᵀ for a [m, k] and b [n, k], as a new [m, n] tensor. It
// packs b on every call; for weights, Pack once and use MatMulPacked.
func MatMulT(a, b *tensor.Tensor) *tensor.Tensor {
	if b.Cols() != a.Cols() {
		panic(fmt.Sprintf("ops.MatMulT: a is %v but b is %v", a.Shape, b.Shape))
	}
	return MatMulPacked(a, Pack(b))
}

// gemm sets c to a·B (plus bias, if not nil, added to every row), where row
// i of a starts at a[i*lda] and c[i][j] is at c[i*ldc+j]; c's prior contents
// do not matter. Strides let attention multiply one head's columns in place.
// Work is split into mc×nc blocks claimed by up to workers goroutines.
func gemm(m int, a []float32, lda int, b *Packed, bias, c []float32, ldc int, workers int) {
	n, k := b.n, b.k
	if m == 0 || n == 0 {
		return
	}
	if bias != nil && len(bias) != n {
		panic(fmt.Sprintf("ops.gemm: bias has %d values for %d columns", len(bias), n))
	}
	if k == 0 {
		for i := range m {
			row := c[i*ldc : i*ldc+n]
			if bias != nil {
				copy(row, bias)
			} else {
				clear(row)
			}
		}
		return
	}
	// Fail on a short view here rather than writing out of bounds in assembly.
	_ = a[(m-1)*lda+k-1]
	_ = c[(m-1)*ldc+n-1]

	// Pack all of a up front: every column block reuses the same strips, and
	// packing costs m·k against the product's m·n·k.
	buf := getScratch((m + mr - 1) / mr * mr * k)
	defer scratch.Put(buf)
	packA(*buf, a, lda, m, k)

	blocksM := (m + mc - 1) / mc
	blocksN := (n + nc - 1) / nc
	forEach(blocksM*blocksN, workers, func(t int) {
		i0, j0 := (t/blocksN)*mc, (t%blocksN)*nc
		gemmBlock(*buf, b, bias, c, ldc, i0, min(i0+mc, m), j0, min(j0+nc, n))
	})
}

// gemmBlock computes rows [i0, i1) and columns [j0, j1) of c from the packed
// strips of a (i0 is a multiple of 2*mr, j0 of 2*nr). It walks each kc-deep
// slice column-panel by column-panel, using the wide kernel for full 12×32
// tiles when it is active and the 6×16 kernel everywhere else.
func gemmBlock(packed []float32, b *Packed, bias, c []float32, ldc, i0, i1, j0, j1 int) {
	for k0 := 0; k0 < b.k; k0 += kc {
		t := tiler{packed: packed, b: b, bias: bias, c: c, ldc: ldc, k0: k0, depth: min(kc, b.k-k0), mode: addC}
		if k0 == 0 {
			t.mode = storeC
			if bias != nil {
				t.mode = storeBiasC
			}
		}
		for j := j0; j < j1; {
			panels := 1
			if wide && j+2*nr <= j1 {
				panels = 2
			}
			if panels == 2 {
				i := i0
				for ; i+2*mr <= i1; i += 2 * mr {
					t.wide(i, j)
				}
				for ; i < i1; i += mr {
					t.tile(i, j, i1, j1)
					t.tile(i, j+nr, i1, j1)
				}
			} else {
				for i := i0; i < i1; i += mr {
					t.tile(i, j, i1, j1)
				}
			}
			j += panels * nr
		}
	}
}

// tiler runs microkernels over one kc-deep slice of a block.
type tiler struct {
	packed  []float32
	b       *Packed
	bias, c []float32
	ldc     int
	k0      int
	depth   int
	mode    int
}

// strip returns the packed slice of a's strip holding row i (a multiple of mr).
func (t *tiler) strip(i int) *float32 {
	return &t.packed[(i/mr)*t.b.k*mr+t.k0*mr]
}

// half reports whether b's panels are fp16, read directly by the kernels.
func (t *tiler) half() bool {
	return t.b.half != nil
}

// panel returns the kc slice of b's float32 panel holding column j (a
// multiple of nr).
func (t *tiler) panel(j int) *float32 {
	return &t.b.data[(j/nr)*t.b.k*nr+t.k0*nr]
}

// panelHalf is panel for fp16 panels.
func (t *tiler) panelHalf(j int) *uint16 {
	return &t.b.half[(j/nr)*t.b.k*nr+t.k0*nr]
}

// kernel runs the 6×16 kernel for the panel storage in use.
func (t *tiler) kernel(i, j int, c *float32, ldc int, bias *float32, mode int) {
	if t.half() {
		kernel6x16Half(t.strip(i), t.panelHalf(j), t.depth, c, ldc, bias, mode)
		return
	}
	kernel6x16(t.strip(i), t.panel(j), t.depth, c, ldc, bias, mode)
}

// wide runs the 12×32 kernel on the full tile at rows i, columns j.
func (t *tiler) wide(i, j int) {
	var biasRow *float32
	if t.mode == storeBiasC {
		biasRow = &t.bias[j]
	}
	c := &t.c[i*t.ldc+j]
	if t.half() {
		kernel12x32Half(t.strip(i), t.strip(i+mr), t.panelHalf(j), t.panelHalf(j+nr), t.depth, c, t.ldc, biasRow, t.mode)
		return
	}
	kernel12x32(t.strip(i), t.strip(i+mr), t.panel(j), t.panel(j+nr), t.depth, c, t.ldc, biasRow, t.mode)
}

// tile runs the 6×16 kernel on the tile at rows i, columns j, clipped to
// rows < i1 and columns < j1. Full tiles are written by the kernel directly;
// edge tiles go through a buffer so only their valid part is stored.
func (t *tiler) tile(i, j, i1, j1 int) {
	rows, cols := min(mr, i1-i), min(nr, j1-j)
	if rows == mr && cols == nr {
		var biasRow *float32
		if t.mode == storeBiasC {
			biasRow = &t.bias[j]
		}
		t.kernel(i, j, &t.c[i*t.ldc+j], t.ldc, biasRow, t.mode)
		return
	}
	var edge [mr * nr]float32
	t.kernel(i, j, &edge[0], nr, nil, storeC)
	for r := range rows {
		dst := t.c[(i+r)*t.ldc+j : (i+r)*t.ldc+j+cols]
		src := edge[r*nr : r*nr+cols]
		switch t.mode {
		case storeC:
			copy(dst, src)
		case storeBiasC:
			for jj, v := range src {
				dst[jj] = v + t.bias[j+jj]
			}
		case addC:
			for jj, v := range src {
				dst[jj] += v
			}
		}
	}
}

// packA copies the m×k matrix a (row i at a[i*lda]) into strips of mr rows,
// each k-major: row r, column kk of strip s lands at dst[s*k*mr + kk*mr + r],
// so a kc-deep slice of a strip is contiguous. Rows past m in the last strip
// are zero.
func packA(dst, a []float32, lda, m, k int) {
	for s := 0; s*mr < m; s++ {
		strip := dst[s*k*mr : (s+1)*k*mr]
		i := s * mr
		if i+mr <= m {
			// Full strip: walk k once, writing each strip position in order.
			r0, r1, r2 := a[i*lda:i*lda+k], a[(i+1)*lda:(i+1)*lda+k], a[(i+2)*lda:(i+2)*lda+k]
			r3, r4, r5 := a[(i+3)*lda:(i+3)*lda+k], a[(i+4)*lda:(i+4)*lda+k], a[(i+5)*lda:(i+5)*lda+k]
			for kk, v0 := range r0 {
				o := strip[kk*mr : kk*mr+mr : kk*mr+mr]
				o[0], o[1], o[2], o[3], o[4], o[5] = v0, r1[kk], r2[kk], r3[kk], r4[kk], r5[kk]
			}
			continue
		}
		for r := range mr {
			if i+r >= m {
				for kk := range k {
					strip[kk*mr+r] = 0
				}
				continue
			}
			for kk, v := range a[(i+r)*lda : (i+r)*lda+k] {
				strip[kk*mr+r] = v
			}
		}
	}
}

// wide selects the 12×32 kernel for full 12-row, 32-column tiles. It is set
// at startup when a wide SIMD kernel passes its self-check; tests also set it
// to exercise the driver with the portable wide kernel.
var wide bool

// workersFor returns how many threads a job of macs multiply-adds deserves.
func workersFor(macs int) int {
	return max(1, min(runtime.GOMAXPROCS(0), macs/macsPerWorker))
}

// forEach runs fn(0) … fn(n-1) on up to workers goroutines. Workers claim
// indexes from a shared counter, so one on a slow core simply takes fewer.
func forEach(n, workers int, fn func(int)) {
	workers = min(workers, n)
	if workers <= 1 {
		for i := range n {
			fn(i)
		}
		return
	}
	var next atomic.Int64
	work := func() {
		for i := int(next.Add(1) - 1); i < n; i = int(next.Add(1) - 1) {
			fn(i)
		}
	}
	var wg sync.WaitGroup
	for range workers - 1 {
		wg.Go(work)
	}
	work()
	wg.Wait()
}
