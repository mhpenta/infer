// Package benchcmp compares infer's matrix product with other Go matrix
// libraries on the shapes BERT models use. It is a separate module so infer never
// depends on them.
//
// Run single-threaded on x86-64 with AVX2:
//
//	GOMAXPROCS=1 go test -run . -bench . -count 10 | tee new.txt
//	benchstat new.txt
//
// Each library is first checked against a float64 reference.
package benchcmp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/viterin/vek/vek32"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// shape is c[m, n] = a[m, k] · w[n, k]ᵀ: m tokens through an nn.Linear with
// k inputs and n outputs.
type shape struct{ m, n, k int }

func (s shape) String() string { return fmt.Sprintf("m%d_n%d_k%d", s.m, s.n, s.k) }

// Token counts from a short query to a full 512-token chunk, against MiniLM
// (hidden 384, FFN 1536) and mxbai (hidden 1024, FFN 4096) weights.
var shapes = func() []shape {
	var out []shape
	for _, m := range []int{3, 29, 128, 512} {
		for _, nk := range [][2]int{{384, 384}, {1536, 384}, {384, 1536}, {4096, 1024}} {
			out = append(out, shape{m, nk[0], nk[1]})
		}
	}
	return out
}()

// operands holds one problem in every layout a library wants. Weights are
// static, so layout conversion happens once, outside the timed loop.
type operands struct {
	shape
	a  []float32 // [m, k]
	w  []float32 // [n, k], nn.Linear layout
	wt []float32 // [k, n], for libraries that only multiply untransposed
	wp *ops.Packed
}

func newOperands(s shape, seed uint64) operands {
	r := rand.New(rand.NewPCG(seed, seed+1))
	o := operands{shape: s, a: make([]float32, s.m*s.k), w: make([]float32, s.n*s.k), wt: make([]float32, s.k*s.n)}
	for i := range o.a {
		o.a[i] = float32(r.NormFloat64())
	}
	for i := range o.w {
		o.w[i] = float32(r.NormFloat64())
	}
	for j := range s.n {
		for p := range s.k {
			o.wt[p*s.n+j] = o.w[j*s.k+p]
		}
	}
	o.wp = ops.Pack(tensor.FromData(o.w, s.n, s.k))
	return o
}

// impl computes a·wᵀ into a fresh [m, n] slice.
type impl struct {
	name string
	run  func(o operands) []float32
}

var gonumBLAS gonum.Implementation

var impls = []impl{
	{"infer", func(o operands) []float32 {
		return ops.MatMulPacked(tensor.FromData(o.a, o.m, o.k), o.wp).Data
	}},
	{"vek", func(o operands) []float32 {
		return vek32.MatMul(o.a, o.wt, o.k)
	}},
	{"gonum", func(o operands) []float32 {
		c := make([]float32, o.m*o.n)
		gonumBLAS.Sgemm(blas.NoTrans, blas.Trans, o.m, o.n, o.k, 1, o.a, o.k, o.w, o.k, 0, c, o.n)
		return c
	}},
}

func TestImplementationsMatchReference(t *testing.T) {
	for i, s := range shapes {
		o := newOperands(s, uint64(i))
		want := make([]float64, s.m*s.n)
		for r := range s.m {
			for c := range s.n {
				var sum float64
				for p := range s.k {
					sum += float64(o.a[r*s.k+p]) * float64(o.w[c*s.k+p])
				}
				want[r*s.n+c] = sum
			}
		}
		tol := 4e-5 * math.Sqrt(float64(s.k))
		for _, im := range impls {
			got := im.run(o)
			for e := range want {
				if math.Abs(float64(got[e])-want[e]) > tol*(1+math.Abs(want[e])) {
					t.Fatalf("%s %v: element %d = %g, want %g", im.name, s, e, got[e], want[e])
				}
			}
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	for i, s := range shapes {
		o := newOperands(s, uint64(i))
		for _, im := range impls {
			b.Run(fmt.Sprintf("%s/%s", s, im.name), func(b *testing.B) {
				for b.Loop() {
					im.run(o)
				}
				b.ReportMetric(2*float64(s.m*s.n*s.k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}
