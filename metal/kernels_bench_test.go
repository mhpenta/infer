package metal

import (
	"fmt"
	"testing"
	"time"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// decodeShapes are MiniCPM5-1B's products in one decoding step: a layer's
// four (q|k|v and gate|up stacked) and the output layer.
var decodeShapes = []struct {
	name string
	n, k int
}{
	{"qkv", 2560, 1536},
	{"out", 1536, 2048},
	{"gate_up", 9216, 1536},
	{"down", 1536, 4608},
	{"lm_head", 130560, 1536},
}

// Benchmarks cycle through distinct copies of the weights, benchBytes in all
// (more than the M2 Max's 48 MB system cache), so every product reads its
// weights from DRAM, as in a real step where every layer's weights differ.
const (
	benchBytes  = 512 << 20
	benchCycles = 4 // passes over the copies per command buffer, to amortize the round trip
)

// reportGemv reports the time per product and the bandwidth of reading
// bytes of weights in that time.
func reportGemv(b *testing.B, products, bytes int) {
	per := b.Elapsed() / time.Duration(b.N*products)
	b.ReportMetric(float64(per.Nanoseconds())/1e3, "µs/product")
	b.ReportMetric(float64(bytes)/per.Seconds()/1e9, "GB/s")
	b.ReportMetric(0, "ns/op")
}

// benchGemvGPU times a gemv instantiation for weights of type t.
func benchGemvGPU(b *testing.B, d *Device, p *Pipeline, rows, groups, n, k int, t WeightType) {
	size := t.Size() * n * k
	copies := max(1, benchBytes/size)
	w, x, y, scale := newBuffer(b, d, copies*size), newBuffer(b, d, 4*k), newBuffer(b, d, 4*n), newBuffer(b, d, 4*n)
	ws, xs := w.Bytes(), Slice[float32](x)
	for i := range ws {
		ws[i] = byte(i % 61) // the contents do not matter, as long as they are finite
	}
	for i := range xs {
		xs[i] = 1
	}
	clear(scale.Bytes())
	for b.Loop() {
		c := d.NewCommandBuffer()
		for range benchCycles {
			for i := range copies {
				gemv(c, p, rows, groups, y.At(0), x.At(0), Weights{W: w.At(i * size), Scale: scale.At(0), Type: t, N: n, K: k})
			}
		}
		if err := c.Wait(); err != nil {
			b.Fatal(err)
		}
	}
	reportGemv(b, benchCycles*copies, size)
}

// BenchmarkGemv compares the GPU's matrix-vector products with the NEON
// ones (all cores) on MiniCPM5-1B's decoding shapes, for bf16 and int8
// weights.
func BenchmarkGemv(b *testing.B) {
	for _, s := range decodeShapes {
		b.Run(s.name+"/metal/bf16", func(b *testing.B) {
			d, k := kernels(b)
			benchGemvGPU(b, d, k.gemv[BF16], gemvRows, gemvGroups, s.n, s.k, BF16)
		})
		b.Run(s.name+"/metal/int8", func(b *testing.B) {
			d, k := kernels(b)
			benchGemvGPU(b, d, k.gemv[Int8], gemvRows, gemvGroups, s.n, s.k, Int8)
		})
		for _, int8 := range []bool{false, true} {
			name, width := s.name+"/neon/bf16", 2
			if int8 {
				name, width = s.name+"/neon/int8", 1
			}
			b.Run(name, func(b *testing.B) {
				copies := max(1, benchBytes/(2*s.n*s.k))
				wt := tensor.New(s.n, s.k)
				for i := range wt.Data {
					wt.Data[i] = float32(i%256) / 256
				}
				packed := make([]*ops.Packed, copies)
				for i := range packed {
					var err error
					if int8 {
						packed[i] = ops.PackInt8(wt)
					} else if packed[i], err = ops.PackBF16(wt); err != nil {
						b.Fatal(err)
					}
				}
				wt = nil
				x, y := tensor.New(1, s.k), tensor.New(1, s.n)
				for b.Loop() {
					for _, p := range packed {
						ops.LinearInto(y, x, p, nil)
					}
				}
				reportGemv(b, copies, width*s.n*s.k)
			})
		}
	}
}

// BenchmarkGemvVariants times each gemv instantiation in kernels/gemv.metal
// (weights, rows per simdgroup, simdgroups per threadgroup).
func BenchmarkGemvVariants(b *testing.B) {
	d := open(b)
	lib, err := d.Compile(kernelSource())
	if err != nil {
		b.Fatal(err)
	}
	defer lib.Release()
	for _, v := range []struct {
		weights      string
		rows, groups int
	}{{"bf16", 2, 4}, {"bf16", 2, 8}, {"bf16", 4, 4}, {"bf16", 4, 8}, {"bf16", 8, 4}, {"int8", 4, 8}, {"int8", 8, 4}} {
		name := fmt.Sprintf("gemv_%s_r%d_sg%d", v.weights, v.rows, v.groups)
		p, err := lib.Pipeline(name)
		if err != nil {
			b.Fatal(err)
		}
		t := map[string]WeightType{"bf16": BF16, "int8": Int8}[v.weights]
		for _, s := range decodeShapes {
			b.Run(name+"/"+s.name, func(b *testing.B) {
				benchGemvGPU(b, d, p, v.rows, v.groups, s.n, s.k, t)
			})
		}
		p.Release()
	}
}

// BenchmarkGemmVariants times the prompt products on MiniCPM5-1B's gate|up
// and down shapes, for a 65-row prompt and a full 512-row chunk: each
// bf16 gemm_mma instantiation in kernels/gemm.metal.
func BenchmarkGemmVariants(b *testing.B) {
	d := open(b)
	lib, err := d.Compile(kernelSource())
	if err != nil {
		b.Fatal(err)
	}
	defer lib.Release()
	type variant struct {
		name string
		run  func(c *CommandBuffer, y, w, x Ref, m, n, k int)
	}
	var variants []variant
	for _, s := range []mmaShape{{64, 64, 16, 2, 4}, {32, 32, 16, 2, 2}, {64, 64, 16, 2, 2}, {128, 64, 16, 4, 4}} {
		p, err := lib.Pipeline(s.name("bf16"))
		if err != nil {
			b.Fatal(err)
		}
		defer p.Release()
		variants = append(variants, variant{s.name("bf16"), func(c *CommandBuffer, y, w, x Ref, m, n, k int) {
			gemmMMA(c, p, s, y, n, x, k, m, Weights{W: w, Type: BF16, N: n, K: k})
		}})
	}
	for _, s := range []struct {
		name    string
		m, n, k int
	}{{"gate_up_65", 65, 9216, 1536}, {"down_65", 65, 1536, 4608}, {"gate_up_512", 512, 9216, 1536}, {"down_512", 512, 1536, 4608}} {
		w, x, y := newBuffer(b, d, 2*s.n*s.k), newBuffer(b, d, 4*s.m*s.k), newBuffer(b, d, 4*s.m*s.n)
		ws, xs := Slice[uint16](w), Slice[float32](x)
		for i := range ws {
			ws[i] = uint16(0x3c00 + i%256)
		}
		for i := range xs {
			xs[i] = 1
		}
		for _, v := range variants {
			b.Run(s.name+"/"+v.name, func(b *testing.B) {
				const reps = 8
				for b.Loop() {
					c := d.NewCommandBuffer()
					for range reps {
						v.run(c, y.At(0), w.At(0), x.At(0), s.m, s.n, s.k)
					}
					if err := c.Wait(); err != nil {
						b.Fatal(err)
					}
				}
				per := b.Elapsed() / time.Duration(b.N*reps)
				b.ReportMetric(float64(per.Nanoseconds())/1e3, "µs/product")
				b.ReportMetric(float64(2*s.m*s.n*s.k)/per.Seconds()/1e9, "GFLOPS")
				b.ReportMetric(0, "ns/op")
			})
		}
	}
}
