package metal

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"testing"
)

const testSource = `
#include <metal_stdlib>
using namespace metal;

kernel void add(device const float *a [[buffer(0)]],
                device const float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant uint &n [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

struct scaleArgs {
    float s;
    uint n;
};

kernel void scale(device float *x [[buffer(0)]],
                  constant scaleArgs &args [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    if (i < args.n) x[i] *= args.s;
}

kernel void empty() {}
`

// open returns the default device, skipping the test without one unless
// INFER_REQUIRE_METAL=1.
func open(t testing.TB) *Device {
	t.Helper()
	d, err := Open()
	if errors.Is(err, ErrUnavailable) {
		if os.Getenv("INFER_REQUIRE_METAL") == "1" {
			t.Fatal("INFER_REQUIRE_METAL=1 but there is no Metal device")
		}
		t.Skip("no Metal device")
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

func pipelines(t testing.TB, d *Device, names ...string) []*Pipeline {
	t.Helper()
	lib, err := d.Compile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	var ps []*Pipeline
	for _, name := range names {
		p, err := lib.Pipeline(name)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(p.Release)
		ps = append(ps, p)
	}
	return ps
}

func newBuffer(t testing.TB, d *Device, n int) *Buffer {
	t.Helper()
	b, err := d.NewBuffer(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	return b
}

func TestOpen(t *testing.T) {
	d := open(t)
	t.Logf("%s: unified memory %v, max buffer %.1f GB, recommended working set %.1f GB",
		d.Name(), d.UnifiedMemory(), float64(d.MaxBufferLength())/1e9, float64(d.RecommendedMaxWorkingSetSize())/1e9)
	if d.Name() == "" || d.MaxBufferLength() <= 0 || d.RecommendedMaxWorkingSetSize() <= 0 {
		t.Errorf("name %q, max buffer %d, working set %d", d.Name(), d.MaxBufferLength(), d.RecommendedMaxWorkingSetSize())
	}
	if !d.UnifiedMemory() {
		t.Error("an Apple silicon GPU reports no unified memory")
	}
}

// TestAdd runs out = a + b on a length that is not a multiple of the
// threadgroup, then, in the same command buffer, out2 = out + a on the
// second half through buffer offsets: the second dispatch must see the
// first's writes. float32 addition is exact IEEE 754 on both sides, so the
// results must be bit-identical to Go's.
func TestAdd(t *testing.T) {
	d := open(t)
	add := pipelines(t, d, "add")[0]
	if add.ThreadExecutionWidth() != 32 || add.MaxThreadsPerThreadgroup() < 256 {
		t.Errorf("execution width %d, max threads %d", add.ThreadExecutionWidth(), add.MaxThreadsPerThreadgroup())
	}
	const n = 1_000_003
	a, b, out, out2 := newBuffer(t, d, 4*n), newBuffer(t, d, 4*n), newBuffer(t, d, 4*n), newBuffer(t, d, 4*n)
	av, bv, ov, o2v := Slice[float32](a), Slice[float32](b), Slice[float32](out), Slice[float32](out2)
	r := rand.New(rand.NewPCG(1, 2))
	for i := range n {
		av[i], bv[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
	}
	const half = n / 2
	cb := d.NewCommandBuffer()
	cb.SetPipeline(add)
	cb.SetBuffer(0, a, 0)
	cb.SetBuffer(1, b, 0)
	cb.SetBuffer(2, out, 0)
	SetValue(cb, 3, uint32(n))
	cb.DispatchThreads(Size{n, 1, 1}, Size{256, 1, 1})
	cb.SetBuffer(0, out, 4*half)
	cb.SetBuffer(1, a, 4*half)
	cb.SetBuffer(2, out2, 4*half)
	SetValue(cb, 3, uint32(n-half))
	cb.DispatchThreads(Size{n - half, 1, 1}, Size{256, 1, 1})
	if err := cb.Wait(); err != nil {
		t.Fatal(err)
	}
	for i := range n {
		want := av[i] + bv[i]
		if math.Float32bits(ov[i]) != math.Float32bits(want) {
			t.Fatalf("out[%d] = %v, want %v", i, ov[i], want)
		}
		if i >= half && math.Float32bits(o2v[i]) != math.Float32bits(want+av[i]) {
			t.Fatalf("out2[%d] = %v, want %v", i, o2v[i], want+av[i])
		}
	}
}

// TestSetValue passes a parameter struct by value.
func TestSetValue(t *testing.T) {
	d := open(t)
	scale := pipelines(t, d, "scale")[0]
	const n = 1000
	x := newBuffer(t, d, 4*n)
	xv := Slice[float32](x)
	for i := range xv {
		xv[i] = float32(i)
	}
	cb := d.NewCommandBuffer()
	cb.SetPipeline(scale)
	cb.SetBuffer(0, x, 0)
	SetValue(cb, 1, struct {
		s float32
		n uint32
	}{0.5, n - 1}) // leave the last element alone
	cb.DispatchThreadgroups(Size{(n + 63) / 64, 1, 1}, Size{64, 1, 1})
	if err := cb.Wait(); err != nil {
		t.Fatal(err)
	}
	for i, v := range xv {
		want := float32(i) * 0.5
		if i == n-1 {
			want = float32(i)
		}
		if v != want {
			t.Fatalf("x[%d] = %v, want %v", i, v, want)
		}
	}
}

func TestCompileErrors(t *testing.T) {
	d := open(t)
	_, err := d.Compile("kernel void broken( { }")
	if err == nil || !strings.HasPrefix(err.Error(), "metal: compiling: ") || len(err.Error()) < len("metal: compiling: ")+10 {
		t.Errorf("Compile of bad source: %v", err)
	}
	lib, err := d.Compile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Release()
	if _, err := lib.Pipeline("missing"); err == nil || !strings.Contains(err.Error(), `"missing"`) {
		t.Errorf("Pipeline of a missing kernel: %v", err)
	}
}

// TestConcurrent records and runs command buffers from several goroutines at
// once, each holding its own thread and autorelease pool.
func TestConcurrent(t *testing.T) {
	d := open(t)
	scale := pipelines(t, d, "scale")[0]
	const workers, rounds, n = 8, 25, 4096
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := range workers {
		x := newBuffer(t, d, 4*n)
		wg.Go(func() {
			xv := Slice[float32](x)
			for i := range xv {
				xv[i] = float32(w)
			}
			for range rounds {
				cb := d.NewCommandBuffer()
				cb.SetPipeline(scale)
				cb.SetBuffer(0, x, 0)
				SetValue(cb, 1, struct {
					s float32
					n uint32
				}{2, n})
				cb.DispatchThreads(Size{n, 1, 1}, Size{256, 1, 1})
				if err := cb.Wait(); err != nil {
					errs <- err
					return
				}
			}
			if want := float32(w) * (1 << rounds); xv[0] != want || xv[n-1] != want {
				errs <- errors.New("wrong result")
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

// BenchmarkRoundTrip measures the fixed cost of running a command buffer:
// one empty dispatch, commit, and wait.
func BenchmarkRoundTrip(b *testing.B) {
	d := open(b)
	empty := pipelines(b, d, "empty")[0]
	for b.Loop() {
		cb := d.NewCommandBuffer()
		cb.SetPipeline(empty)
		cb.DispatchThreadgroups(Size{1, 1, 1}, Size{32, 1, 1})
		if err := cb.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}
