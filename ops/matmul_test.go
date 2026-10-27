package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

// refGemmT is the ground-truth a·bᵀ: a plain triple loop accumulated in
// float64, on strided views (row i of a at a[i*lda], row j of b at b[j*ldb]).
func refGemmT(m, n, k int, a []float32, lda int, b []float32, ldb int) []float64 {
	c := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var s float64
			for p := range k {
				s += float64(a[i*lda+p]) * float64(b[j*ldb+p])
			}
			c[i*n+j] = s
		}
	}
	return c
}

func randomSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}

// checkClose compares against the float64 reference with a tolerance that
// grows with k, since float32 rounding error accumulates along the sum.
func checkClose(t *testing.T, got []float32, ldc int, want []float64, m, n, k int) {
	t.Helper()
	tol := 1e-5 * math.Sqrt(float64(max(k, 1))) * 4
	for i := range m {
		for j := range n {
			g, w := float64(got[i*ldc+j]), want[i*n+j]
			if math.Abs(g-w) > tol*(1+math.Abs(w)) {
				t.Fatalf("c[%d][%d] = %g, want %g (m=%d n=%d k=%d)", i, j, g, w, m, n, k)
			}
		}
	}
}

func TestMatMulTMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	// Sizes around the 6×16 microkernel, the 72×128 blocks and the 256-deep
	// k blocks.
	ms := []int{1, 2, 5, 6, 7, 12, 71, 72, 73, 150}
	ns := []int{1, 3, 15, 16, 17, 33, 127, 128, 129, 300}
	ks := []int{1, 3, 8, 33, 255, 256, 257, 600}
	for _, m := range ms {
		for _, n := range ns {
			for _, k := range ks {
				a, b := randomSlice(r, m*k), randomSlice(r, n*k)
				got := MatMulT(tensor.FromData(a, m, k), tensor.FromData(b, n, k))
				checkClose(t, got.Data, n, refGemmT(m, n, k, a, k, b, k), m, n, k)
			}
		}
	}
}

func TestGemmParallelMatchesSerial(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	m, n, k := 150, 300, 300
	a, b := randomSlice(r, m*k), randomSlice(r, n*k)
	p := packT(b, n, k, k)
	serial, parallel := make([]float32, m*n), make([]float32, m*n)
	gemm(m, a, k, p, nil, serial, n, 1)
	gemm(m, a, k, p, nil, parallel, n, 4)
	for i := range serial {
		if serial[i] != parallel[i] {
			t.Fatalf("element %d: serial %g, parallel %g", i, serial[i], parallel[i])
		}
	}
}

func TestGemmStridedViews(t *testing.T) {
	// One "head" of width 8 at column offset 16 inside rows of width 40, as
	// attention uses it; the output lands in a wider matrix too.
	r := rand.New(rand.NewPCG(5, 6))
	const width, off, k = 40, 16, 8
	for _, tt := range []struct{ m, n int }{{1, 1}, {5, 3}, {17, 66}} {
		t.Run(fmt.Sprintf("T_%dx%d", tt.m, tt.n), func(t *testing.T) {
			a, b := randomSlice(r, tt.m*width), randomSlice(r, tt.n*width)
			ldc := tt.n + 7
			c := make([]float32, tt.m*ldc)
			for i := range c {
				c[i] = -999 // overwritten inside the columns, kept outside
			}
			gemm(tt.m, a[off:], width, packT(b[off:], tt.n, k, width), nil, c, ldc, 1)
			checkClose(t, c, ldc, refGemmT(tt.m, tt.n, k, a[off:], width, b[off:], width), tt.m, tt.n, k)
			for i := range tt.m {
				for j := tt.n; j < ldc; j++ {
					if c[i*ldc+j] != -999 {
						t.Fatalf("gemm wrote outside its columns at [%d][%d]", i, j)
					}
				}
			}
		})
	}
}

func TestPackN(t *testing.T) {
	// a·b with b given as k rows of n values at a column offset, as attention
	// multiplies probabilities by one head of v.
	r := rand.New(rand.NewPCG(7, 8))
	const width, off = 50, 9
	for _, tt := range []struct{ m, n, k int }{{1, 1, 1}, {7, 17, 5}, {20, 32, 300}} {
		t.Run(fmt.Sprintf("%dx%dx%d", tt.m, tt.n, tt.k), func(t *testing.T) {
			a, b := randomSlice(r, tt.m*tt.k), randomSlice(r, tt.k*width)
			// The same product as a·bᵀ with b transposed to n rows of k.
			bt := make([]float32, tt.n*tt.k)
			for kk := range tt.k {
				for j := range tt.n {
					bt[j*tt.k+kk] = b[kk*width+off+j]
				}
			}
			c := make([]float32, tt.m*tt.n)
			bp := packNInto(make([]float32, packedSize(tt.n, tt.k)), b[off:], tt.k, tt.n, width)
			gemm(tt.m, a, tt.k, &bp, nil, c, tt.n, 1)
			checkClose(t, c, tt.n, refGemmT(tt.m, tt.n, tt.k, a, tt.k, bt, tt.k), tt.m, tt.n, tt.k)
		})
	}
}

// TestKernelMatchesPortable checks the active microkernel (SIMD where there
// is one) against the portable Go kernel directly, in every epilogue mode,
// writing into a strided c.
func TestKernelMatchesPortable(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 10))
	const ldc = nr + 5
	for _, k := range []int{1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 1029} {
		a, b := randomSlice(r, k*mr), randomSlice(r, k*nr)
		bias := randomSlice(r, nr)
		start := randomSlice(r, mr*ldc)
		for _, mode := range []int{storeC, storeBiasC, addC} {
			got, want := slices.Clone(start), slices.Clone(start)
			kernel6x16(&a[0], &b[0], k, &got[0], ldc, &bias[0], mode)
			kernel6x16Portable(&a[0], &b[0], k, &want[0], ldc, &bias[0], mode)
			for i := range got {
				if math.Abs(float64(got[i]-want[i])) > 1e-4*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%s kernel, k=%d, mode %d: c[%d] = %g, portable kernel gives %g", Kernel(), k, mode, i, got[i], want[i])
				}
			}
		}
	}
}

func TestLinearBiasMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(15, 16))
	// The bias is added on the first k block only, by the kernel for full
	// tiles and by Go for edges; k > kc checks it is not added twice.
	for _, m := range []int{1, 6, 7, 73} {
		for _, n := range []int{1, 16, 17, 129} {
			for _, k := range []int{1, 256, 300} {
				x, w, b := randomSlice(r, m*k), randomSlice(r, n*k), randomSlice(r, n)
				got := Linear(tensor.FromData(x, m, k), Pack(tensor.FromData(w, n, k)), tensor.FromData(b, n))
				want := refGemmT(m, n, k, x, k, w, k)
				for i := range m {
					for j := range n {
						want[i*n+j] += float64(b[j])
					}
				}
				checkClose(t, got.Data, n, want, m, n, k)
			}
		}
	}
}

func TestMatMulTZeroK(t *testing.T) {
	c := MatMulT(tensor.New(2, 0), tensor.New(3, 0))
	for _, v := range c.Data {
		if v != 0 {
			t.Fatalf("got %v, want zeros", c.Data)
		}
	}
}

// refAttention is unmasked multi-head attention written directly from the
// definition, accumulated in float64.
func refAttention(q, k, v *tensor.Tensor, heads int, scale float64) []float64 {
	seq, kvSeq, hidden := q.Rows(), k.Rows(), q.Cols()
	dh := hidden / heads
	out := make([]float64, seq*hidden)
	for h := range heads {
		off := h * dh
		for i := range seq {
			scores := make([]float64, kvSeq)
			peak := math.Inf(-1)
			for j := range kvSeq {
				for d := range dh {
					scores[j] += float64(q.Row(i)[off+d]) * float64(k.Row(j)[off+d])
				}
				scores[j] *= scale
				peak = max(peak, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - peak)
				sum += scores[j]
			}
			for j := range kvSeq {
				for d := range dh {
					out[i*hidden+off+d] += scores[j] / sum * float64(v.Row(j)[off+d])
				}
			}
		}
	}
	return out
}

func TestAttentionMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	tests := []struct{ seq, kvSeq, heads, headDim int }{
		{1, 1, 1, 4},
		{3, 3, 12, 32},
		{5, 9, 2, 6},
		{70, 70, 4, 16},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("seq%d_kv%d_h%d_d%d", tt.seq, tt.kvSeq, tt.heads, tt.headDim), func(t *testing.T) {
			hidden := tt.heads * tt.headDim
			q := tensor.FromData(randomSlice(r, tt.seq*hidden), tt.seq, hidden)
			k := tensor.FromData(randomSlice(r, tt.kvSeq*hidden), tt.kvSeq, hidden)
			v := tensor.FromData(randomSlice(r, tt.kvSeq*hidden), tt.kvSeq, hidden)
			scale := 1 / math.Sqrt(float64(tt.headDim))
			got := Attention(q, k, v, tt.heads, float32(scale))
			want := refAttention(q, k, v, tt.heads, scale)
			for i, w := range want {
				if math.Abs(float64(got.Data[i])-w) > 1e-5*(1+math.Abs(w)) {
					t.Fatalf("element %d = %g, want %g", i, got.Data[i], w)
				}
			}
		})
	}
}

// Shapes from MiniLM (hidden 384, FFN 1536) and mxbai (1024, 4096), with
// weights packed once outside the timed loop as a model does.
func BenchmarkMatMulPacked(b *testing.B) {
	shapes := []struct{ m, n, k int }{
		{3, 384, 384},
		{29, 1536, 384},
		{65, 384, 384},
		{65, 1536, 384},
		{65, 384, 1536},
		{512, 1536, 384},
		{128, 4096, 1024},
	}
	r := rand.New(rand.NewPCG(13, 14))
	for _, s := range shapes {
		a := tensor.FromData(randomSlice(r, s.m*s.k), s.m, s.k)
		w := Pack(tensor.FromData(randomSlice(r, s.n*s.k), s.n, s.k))
		b.Run(fmt.Sprintf("%dx%dx%d", s.m, s.n, s.k), func(b *testing.B) {
			for b.Loop() {
				MatMulPacked(a, w)
			}
			b.ReportMetric(2*float64(s.m*s.n*s.k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}

// BenchmarkPanelFormats compares float32, bf16 and int8 panels on
// MiniCPM5-2B's shapes (the stacked gate/up projection and the down
// projection), for decoding (one row) and a 64-token prompt.
func BenchmarkPanelFormats(b *testing.B) {
	r := rand.New(rand.NewPCG(15, 16))
	for _, s := range []struct{ n, k int }{{12288, 2048}, {2048, 6144}} {
		w := narrowWeights(r, formatBF16, s.n, s.k)
		bf16, err := PackBF16(w)
		if err != nil {
			b.Fatal(err)
		}
		for _, f := range []struct {
			name string
			p    *Packed
		}{{"f32", Pack(w)}, {"bf16", bf16}, {"int8", PackInt8(w)}} {
			for _, m := range []int{1, 64} {
				x := tensor.FromData(randomSlice(r, m*s.k), m, s.k)
				y := tensor.New(m, s.n)
				b.Run(fmt.Sprintf("%dx%dx%d/%s", m, s.n, s.k, f.name), func(b *testing.B) {
					for b.Loop() {
						LinearInto(y, x, f.p, nil)
					}
					b.ReportMetric(2*float64(m*s.n*s.k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
				})
			}
		}
	}
}

// TestWideDriverMatchesReference turns on the 12×32 tile path. Here it runs
// the portable wide kernel (or AVX-512 where the self-check enabled it), so
// the driver's mix of wide and 6×16 tiles is tested on every machine.
func TestWideDriverMatchesReference(t *testing.T) {
	saved := wide
	wide = true
	defer func() { wide = saved }()
	r := rand.New(rand.NewPCG(17, 18))
	for _, m := range []int{1, 11, 12, 13, 24, 30, 73, 150} {
		for _, n := range []int{16, 31, 32, 33, 64, 100, 129} {
			for _, k := range []int{1, 7, 256, 300} {
				x, w, b := randomSlice(r, m*k), randomSlice(r, n*k), randomSlice(r, n)
				got := Linear(tensor.FromData(x, m, k), Pack(tensor.FromData(w, n, k)), tensor.FromData(b, n))
				want := refGemmT(m, n, k, x, k, w, k)
				for i := range m {
					for j := range n {
						want[i*n+j] += float64(b[j])
					}
				}
				checkClose(t, got.Data, n, want, m, n, k)
			}
		}
	}
}

func TestAttentionQKVMatchesSeparate(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 20))
	const seq, heads, headDim = 23, 4, 8
	hidden := heads * headDim
	qkv := tensor.FromData(randomSlice(r, seq*3*hidden), seq, 3*hidden)
	split := func(part int) *tensor.Tensor {
		t := tensor.New(seq, hidden)
		for i := range seq {
			copy(t.Row(i), qkv.Row(i)[part*hidden:(part+1)*hidden])
		}
		return t
	}
	want := Attention(split(0), split(1), split(2), heads, 0.35)
	got := tensor.New(seq, hidden)
	AttentionQKVInto(got, qkv, heads, 0.35)
	nearAll(t, got.Data, want.Data)
}

func TestPackA(t *testing.T) {
	r := rand.New(rand.NewPCG(27, 28))
	for _, m := range []int{1, 5, 6, 7, 13} {
		const k, lda = 9, 12
		a := randomSlice(r, m*lda)
		strips := (m + mr - 1) / mr
		dst := randomSlice(r, strips*mr*k) // packA must overwrite everything
		packA(dst, a, lda, m, k)
		for s := range strips {
			for kk := range k {
				for rr := range mr {
					want := float32(0)
					if i := s*mr + rr; i < m {
						want = a[i*lda+kk]
					}
					if got := dst[s*k*mr+kk*mr+rr]; got != want {
						t.Fatalf("m=%d strip %d k %d row %d: %g, want %g", m, s, kk, rr, got, want)
					}
				}
			}
		}
	}
}

func BenchmarkPackA(b *testing.B) {
	for _, s := range []struct{ m, k int }{{65, 384}, {65, 1536}, {512, 384}} {
		r := rand.New(rand.NewPCG(29, 30))
		a := randomSlice(r, s.m*s.k)
		dst := make([]float32, (s.m+mr-1)/mr*mr*s.k)
		b.Run(fmt.Sprintf("%dx%d", s.m, s.k), func(b *testing.B) {
			for b.Loop() {
				packA(dst, a, s.k, s.m, s.k)
			}
		})
	}
}
