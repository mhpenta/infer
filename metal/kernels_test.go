package metal

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/mhpenta/infer/internal/half"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

func kernels(t testing.TB) (*Device, *Kernels) {
	t.Helper()
	d := open(t)
	k, err := d.Kernels()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(k.Release)
	return d, k
}

// upload returns a buffer holding pad zero bytes, then v.
func upload[T float32 | uint16 | int32 | int8](t testing.TB, d *Device, pad int, v []T) *Buffer {
	t.Helper()
	var zero T
	b := newBuffer(t, d, pad+len(v)*int(unsafe.Sizeof(zero)))
	clear(b.Bytes())
	copy(Slice[T](&Buffer{data: b.Bytes()[pad:]}), v)
	return b
}

// read returns n values of T at byte offset off of b.
func read[T float32 | int32](b *Buffer, off, n int) []T {
	return slices.Clone(Slice[T](&Buffer{data: b.Bytes()[off:]})[:n])
}

func run(t testing.TB, d *Device, record func(c *CommandBuffer)) {
	t.Helper()
	c := d.NewCommandBuffer()
	record(c)
	if err := c.Wait(); err != nil {
		t.Fatal(err)
	}
}

func normals(r *rand.Rand, n int, scale float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64() * scale)
	}
	return v
}

// bf16s returns n bf16 values (truncated normals) and their float32 values.
func bf16s(r *rand.Rand, n int, scale float64) ([]uint16, []float32) {
	h, f := make([]uint16, n), make([]float32, n)
	for i := range h {
		h[i] = uint16(math.Float32bits(float32(r.NormFloat64()*scale)) >> 16)
		f[i] = math.Float32frombits(uint32(h[i]) << 16)
	}
	return h, f
}

// within fails unless |got-want| ≤ rel·|want| + abs for every element, and
// returns the largest error relative to the largest |want|.
func within(t *testing.T, name string, got, want []float32, rel, abs float64) float64 {
	t.Helper()
	var worst, peak float64
	for i := range want {
		d, w := math.Abs(float64(got[i])-float64(want[i])), math.Abs(float64(want[i]))
		if !(d <= rel*w+abs) {
			t.Fatalf("%s[%d] = %v, want %v (error %.3g)", name, i, got[i], want[i], d)
		}
		worst, peak = max(worst, d), max(peak, w)
	}
	return worst / peak
}

func exact[T float32 | int32](t *testing.T, name string, got, want []T) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] && !(got[i] != got[i] && want[i] != want[i]) {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// TestGemvBF16 checks the product against a float64 reference, within a
// bound on float32 rounding (relative to Σ|w·x|, which bounds any summation
// order's error), on model shapes and on ones that are not multiples of the
// kernel's tiles, with the weights at an 8-byte (not 16-byte) offset. The
// NEON kernel is held to the same bound, for comparison.
func TestGemvBF16(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(1, 2))
	for _, sh := range [][2]int{{2560, 1536}, {1536, 2048}, {9216, 1536}, {37, 100}, {5, 260}, {1, 4}, {131, 1028}} {
		n, kd := sh[0], sh[1]
		wh, wf := bf16s(r, n*kd, 0.05)
		x := normals(r, kd, 1)
		w, xb, y := upload(t, d, 8, wh), upload(t, d, 16, x), newBuffer(t, d, 4+4*n)
		run(t, d, func(c *CommandBuffer) {
			k.Linear(c, y.At(4), n, xb.At(16), kd, 1, Weights{W: w.At(8), Type: BF16, N: n, K: kd})
		})
		got := read[float32](y, 4, n)

		wt := tensor.FromData(wf, n, kd)
		p, err := ops.PackBF16(wt)
		if err != nil {
			t.Fatal(err)
		}
		cpu := tensor.New(1, n)
		ops.LinearInto(cpu, tensor.FromData(x, 1, kd), p, nil)

		var worstGPU, worstCPU float64
		for i := range n {
			var sum, mag float64
			for j := range kd {
				prod := float64(wf[i*kd+j]) * float64(x[j])
				sum, mag = sum+prod, mag+math.Abs(prod)
			}
			bound := float64(kd) * 0x1p-24 * mag
			eg, ec := math.Abs(float64(got[i])-sum), math.Abs(float64(cpu.Data[i])-sum)
			if eg > bound {
				t.Fatalf("%dx%d: y[%d] = %v, want %v (error %.3g, bound %.3g)", n, kd, i, got[i], sum, eg, bound)
			}
			if ec > bound {
				t.Errorf("%dx%d: NEON y[%d] = %v, want %v (error %.3g, bound %.3g)", n, kd, i, cpu.Data[i], sum, ec, bound)
			}
			worstGPU, worstCPU = max(worstGPU, eg/mag), max(worstCPU, ec/mag)
		}
		t.Logf("%5dx%-5d largest error / Σ|w·x|: GPU %.2g, CPU %.2g", n, kd, worstGPU, worstCPU)
	}
}

// TestRMSNorm runs a layer's norm (one group per row, into a separate
// buffer) and Qwen3's per-head norms (in place, on the query and key parts of
// stacked q|k|v rows) against ops.RMSNorm, which sums in float64.
func TestRMSNorm(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(3, 4))
	const eps = 1e-6
	check := func(name string, rows, ld, off, groups, dim int, inPlace bool) {
		x := normals(r, rows*ld, 2)
		w := normals(r, dim, 1)
		src, wb := upload(t, d, 0, x), upload(t, d, 0, w)
		dst := src
		if !inPlace {
			dst = newBuffer(t, d, 4*rows*ld)
			copy(Slice[float32](dst), x)
		}
		run(t, d, func(c *CommandBuffer) {
			k.RMSNorm(c, dst.At(4*off), src.At(4*off), wb.At(0), ld, rows, groups, dim, eps)
		})
		want := slices.Clone(x)
		for i := range rows {
			ops.RMSNorm(tensor.FromData(want[i*ld+off:i*ld+off+groups*dim], groups, dim), tensor.FromData(w, dim), eps)
		}
		t.Logf("%s: largest error / largest value %.2g", name, within(t, name, read[float32](dst, 0, rows*ld), want, 1e-6, 1e-7))
	}
	check("layer norm", 3, 1536, 0, 1, 1536, false)
	check("layer norm, odd width", 2, 1000, 0, 1, 1000, true)
	check("query heads", 4, 2560, 0, 16, 128, true)
	check("key heads", 4, 2560, 2048, 2, 128, true)
}

// TestRoPE rotates the query and key heads of stacked q|k|v rows in one call
// and checks the result is bit-identical to ops.Rotation.Apply, with the
// value part untouched.
func TestRoPE(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(5, 6))
	const heads, kvHeads, dim, rows, pos = 16, 2, 128, 5, 2457
	ld := (heads + 2*kvHeads) * dim
	x := normals(r, rows*ld, 1)
	rot := ops.NewRoPE(dim, 5e6).Rotation(pos, rows)
	cos, sin := rot.Tables()
	xb, cb, sb := upload(t, d, 0, x), upload(t, d, 0, cos), upload(t, d, 0, sin)
	run(t, d, func(c *CommandBuffer) { k.RoPE(c, xb.At(0), cb.At(0), sb.At(0), ld, rows, heads+kvHeads, dim) })
	want := slices.Clone(x)
	rot.Apply(want, ld, heads)
	rot.Apply(want[heads*dim:], ld, kvHeads)
	exact(t, "x", read[float32](xb, 0, rows*ld), want)
}

func TestSwiGLU(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(7, 8))
	const rows, ff = 3, 4608
	gu := normals(r, rows*2*ff, 3)
	in, out := upload(t, d, 0, gu), newBuffer(t, d, 4*rows*ff)
	run(t, d, func(c *CommandBuffer) { k.SwiGLU(c, out.At(0), in.At(0), rows, ff) })
	want := tensor.New(rows, ff)
	ops.SwiGLUInto(want, tensor.FromData(gu, rows, 2*ff))
	t.Logf("largest error / largest value %.2g", within(t, "out", read[float32](out, 0, rows*ff), want.Data, 1e-6, 1e-12))
}

func TestAddKernel(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(9, 10))
	const n = 1536*3 + 7
	a, b := normals(r, n, 1), normals(r, n, 1)
	ab, bb := upload(t, d, 0, a), upload(t, d, 0, b)
	run(t, d, func(c *CommandBuffer) { k.Add(c, ab.At(0), bb.At(0), n) })
	want := slices.Clone(a)
	ops.Add(want, b)
	exact(t, "dst", read[float32](ab, 0, n), want)
}

func TestEmbed(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(11, 12))
	const vocab, cols = 1000, 1536
	th, tf := bf16s(r, vocab*cols, 0.02)
	ids := []int32{0, 999, 17, 17, 512}
	table, ib, out := upload(t, d, 0, th), upload(t, d, 0, ids), newBuffer(t, d, 4*len(ids)*cols)
	run(t, d, func(c *CommandBuffer) { k.Embed(c, out.At(0), table.At(0), BF16, ib.At(0), vocab, len(ids), cols) })
	var want []float32
	for _, id := range ids {
		want = append(want, tf[int(id)*cols:int(id+1)*cols]...)
	}
	exact(t, "out", read[float32](out, 0, len(want)), want)
}

func TestArgmax(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(13, 14))
	inf := float32(math.Inf(-1))
	random := normals(r, 130560, 3)
	tie := normals(r, 151936, 1)
	tie[70000], tie[150000], tie[90000] = 50, 50, 50
	last := normals(r, 1500, 1)
	last[1499] = 9
	for name, x := range map[string][]float32{
		"random": random, "tie": tie, "last": last,
		"one": {-3}, "equal": {2, 2, 2, 2}, "all -Inf": {inf, inf, inf},
	} {
		xb, out := upload(t, d, 0, x), newBuffer(t, d, 4)
		run(t, d, func(c *CommandBuffer) { k.Argmax(c, out.At(0), xb.At(0), len(x)) })
		if got, want := read[int32](out, 0, 1)[0], int32(ops.Argmax(x)); got != want {
			t.Errorf("%s: argmax %d, want %d", name, got, want)
		}
	}
}

// TestAttend appends keys and values to a GPU cache and attends to it, and
// does the same with ops.KVCache: a first token, decoding steps after short
// and long prefixes (the one-row kernel), and causal runs of rows (the tiled
// kernel) from a few to a 512-row chunk, after prefixes that do and don't
// line up with its tiles, with grouped-query attention as in MiniCPM5 (16
// query heads over 2 key/value heads) and without.
func TestAttend(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(15, 16))
	for _, tc := range []struct {
		name                string
		heads, kvHeads, dim int
		prefix, rows        int
	}{
		{"first token", 16, 2, 128, 0, 1},
		{"decode", 16, 2, 128, 300, 1},
		{"long decode", 16, 2, 128, 2500, 1},
		{"causal rows", 16, 2, 128, 40, 7},
		{"no grouping", 4, 4, 64, 100, 3},
		{"short heads", 8, 2, 32, 9, 2},
		{"prompt", 16, 2, 128, 0, 300},
		{"after prefix", 16, 2, 128, 37, 100},
		{"long chunk", 16, 2, 128, 1500, 512},
		{"prompt, 64", 8, 8, 64, 5, 70},
	} {
		qDim, kvDim := tc.heads*tc.dim, tc.kvHeads*tc.dim
		width := qDim + 2*kvDim
		total := tc.prefix + tc.rows
		qkv := normals(r, total*width, 1)
		scale := float32(1 / math.Sqrt(float64(tc.dim)))

		cpu := ops.NewKVCache(tc.kvHeads, tc.dim, total)
		cpu.Append(qkv[qDim:], width, qkv[qDim+kvDim:], width, total)
		want := make([]float32, tc.rows*qDim)
		cpu.Attend(want, qDim, qkv[tc.prefix*width:], width, tc.rows, ops.AttentionArgs{
			Heads: tc.heads, KVHeads: tc.kvHeads, Scale: scale, Causal: true, Offset: tc.prefix,
		})

		kv, err := d.NewKVCache(tc.kvHeads, tc.dim, total+5)
		if err != nil {
			t.Fatal(err)
		}
		defer kv.Release()
		in, out := upload(t, d, 0, qkv), newBuffer(t, d, 4*tc.rows*qDim)
		run(t, d, func(c *CommandBuffer) {
			if tc.prefix > 0 {
				k.KVAppend(c, kv, in.At(4*qDim), in.At(4*(qDim+kvDim)), width, tc.prefix, 0)
			}
			rows := in.At(4 * tc.prefix * width)
			k.KVAppend(c, kv, Ref{in, rows.Off + 4*qDim}, Ref{in, rows.Off + 4*(qDim+kvDim)}, width, tc.rows, tc.prefix)
			k.Attend(c, out.At(0), qDim, rows, width, kv, tc.rows, AttendArgs{Heads: tc.heads, Scale: scale, Causal: true, Offset: tc.prefix})
		})
		// The cache must hold exactly the appended keys and values.
		keys, values := read[float32](kv.Keys, 0, tc.kvHeads*kv.Capacity*tc.dim), read[float32](kv.Values, 0, tc.kvHeads*kv.Capacity*tc.dim)
		for pos := range total {
			for h := range tc.kvHeads {
				at := (h*kv.Capacity + pos) * tc.dim
				row := qkv[pos*width+qDim+h*tc.dim:]
				exact(t, tc.name+" keys", keys[at:at+tc.dim], row[:tc.dim])
				exact(t, tc.name+" values", values[at:at+tc.dim], row[kvDim:kvDim+tc.dim])
			}
		}
		t.Logf("%-12s largest error / largest value %.2g", tc.name, within(t, tc.name, read[float32](out, 0, len(want)), want, 2e-6, 2e-6))
	}
}

// TestGemmBF16 checks the simdgroup-matrix product against a float64
// reference, within the same bound as TestGemvBF16, on prompt-sized shapes,
// edges that cut through every tile, a k that is not a multiple of the
// kernel's step, strided rows, and one row.
func TestGemmBF16(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(19, 20))
	for _, sh := range [][3]int{{65, 2560, 1536}, {130, 1536, 4608}, {7, 37, 100}, {100, 131, 1028}, {1, 9216, 1536}, {64, 64, 16}} {
		m, n, kd := sh[0], sh[1], sh[2]
		ldx, ldy := kd+4, n+3
		wh, wf := bf16s(r, n*kd, 0.05)
		x := normals(r, m*ldx, 1)
		w, xb, y := upload(t, d, 8, wh), upload(t, d, 0, x), newBuffer(t, d, 4*m*ldy)
		run(t, d, func(c *CommandBuffer) {
			k.Linear(c, y.At(0), ldy, xb.At(0), ldx, m, Weights{W: w.At(8), Type: BF16, N: n, K: kd})
		})
		got := read[float32](y, 0, m*ldy)
		var worst float64
		for i := range m {
			for j := range n {
				var sum, mag float64
				for l := range kd {
					p := float64(wf[j*kd+l]) * float64(x[i*ldx+l])
					sum, mag = sum+p, mag+math.Abs(p)
				}
				e := math.Abs(float64(got[i*ldy+j]) - sum)
				if e > float64(kd)*0x1p-24*mag {
					t.Fatalf("%dx%dx%d: y[%d,%d] = %v, want %v (error %.3g)", m, n, kd, i, j, got[i*ldy+j], sum, e)
				}
				worst = max(worst, e/mag)
			}
		}
		t.Logf("%4dx%-5dx%-5d largest error / Σ|w·x|: %.2g", m, n, kd, worst)
	}
}

// int8Weights returns an n×k weight of normal draws with an outlier in every
// third row, so that rows get very different scales, quantized by
// ops.QuantizeInt8.
func int8Weights(r *rand.Rand, n, k int) (*tensor.Tensor, []int8, []float32) {
	w := tensor.FromData(normals(r, n*k, 0.05), n, k)
	for j := 0; j < n; j += 3 {
		w.Row(j)[r.IntN(k)] *= 20
	}
	q, scale := ops.QuantizeInt8(w)
	return w, q, scale
}

// checkInt8 checks rows of y against y[i][j] = scale[j]·Σ q[j,k]·x[i,k] in
// float64, within float32 rounding of the sum and of the scaling, and
// returns the largest error relative to scale[j]·Σ|q·x|.
func checkInt8(t *testing.T, name string, got []float32, ldy int, q []int8, scale []float32, x []float32, ldx, m, n, kd int) float64 {
	t.Helper()
	var worst float64
	for i := range m {
		for j := range n {
			var sum, mag float64
			for l := range kd {
				p := float64(q[j*kd+l]) * float64(x[i*ldx+l])
				sum, mag = sum+p, mag+math.Abs(p)
			}
			s := float64(scale[j])
			want := sum * s
			e := math.Abs(float64(got[i*ldy+j]) - want)
			if e > (float64(kd)*mag*s+math.Abs(want))*0x1p-24 {
				t.Fatalf("%s: y[%d,%d] = %v, want %v (error %.3g)", name, i, j, got[i*ldy+j], want, e)
			}
			if mag > 0 {
				worst = max(worst, e/(mag*s))
			}
		}
	}
	return worst
}

// TestGemvInt8 checks the int8 one-row product against float64 on
// MiniCPM5-1B's shapes and on ones that are not multiples of the kernel's
// tiles, and against ops.PackInt8's product on the CPU.
func TestGemvInt8(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(21, 22))
	for _, sh := range [][2]int{{2560, 1536}, {1536, 4608}, {37, 100}, {5, 260}, {1, 4}} {
		n, kd := sh[0], sh[1]
		wt, q, scale := int8Weights(r, n, kd)
		x := normals(r, kd, 1)
		w, sb, xb, y := upload(t, d, 4, q), upload(t, d, 0, scale), upload(t, d, 16, x), newBuffer(t, d, 4*n)
		run(t, d, func(c *CommandBuffer) {
			k.Linear(c, y.At(0), n, xb.At(16), kd, 1, Weights{W: w.At(4), Scale: sb.At(0), Type: Int8, N: n, K: kd})
		})
		name := fmt.Sprintf("%dx%d", n, kd)
		got := read[float32](y, 0, n)
		worst := checkInt8(t, name, got, n, q, scale, x, kd, 1, n, kd)
		cpu := tensor.New(1, n)
		ops.LinearInto(cpu, tensor.FromData(x, 1, kd), ops.PackInt8(wt), nil)
		cpuWorst := checkInt8(t, name+" NEON", cpu.Data, n, q, scale, x, kd, 1, n, kd)
		t.Logf("%5dx%-5d largest error / scale·Σ|q·x|: GPU %.2g, CPU %.2g", n, kd, worst, cpuWorst)
	}
}

// TestGemmInt8 checks the int8 simdgroup-matrix product as TestGemmBF16
// checks the bf16 one.
func TestGemmInt8(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(23, 24))
	for _, sh := range [][3]int{{65, 2560, 1536}, {7, 37, 100}, {100, 131, 1028}, {1, 9216, 1536}, {64, 64, 16}} {
		m, n, kd := sh[0], sh[1], sh[2]
		ldx, ldy := kd+4, n+3
		_, q, scale := int8Weights(r, n, kd)
		x := normals(r, m*ldx, 1)
		w, sb, xb, y := upload(t, d, 4, q), upload(t, d, 0, scale), upload(t, d, 0, x), newBuffer(t, d, 4*m*ldy)
		run(t, d, func(c *CommandBuffer) {
			k.Linear(c, y.At(0), ldy, xb.At(0), ldx, m, Weights{W: w.At(4), Scale: sb.At(0), Type: Int8, N: n, K: kd})
		})
		worst := checkInt8(t, fmt.Sprintf("%dx%dx%d", m, n, kd), read[float32](y, 0, m*ldy), ldy, q, scale, x, ldx, m, n, kd)
		t.Logf("%4dx%-5dx%-5d largest error / scale·Σ|q·x|: %.2g", m, n, kd, worst)
	}
}

// TestLinearTypes checks float32 and fp16 weights, with and without a bias,
// through the one-row and the many-row kernels, against float64.
func TestLinearTypes(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(25, 26))
	for _, typ := range []WeightType{F32, F16, BF16} {
		for _, biased := range []bool{false, true} {
			for _, m := range []int{1, 70} {
				const n, kd = 131, 388
				wf := normals(r, n*kd, 0.05)
				var raw []byte
				switch typ {
				case F32:
					raw = append(raw, unsafe.Slice((*byte)(unsafe.Pointer(&wf[0])), 4*len(wf))...)
				case F16:
					for i := range wf {
						h := fp16(r)
						wf[i] = half.ToFloat32(h)
						raw = append(raw, byte(h), byte(h>>8))
					}
				case BF16:
					for i, v := range wf {
						h := uint16(math.Float32bits(v) >> 16)
						wf[i] = math.Float32frombits(uint32(h) << 16)
						raw = append(raw, byte(h), byte(h>>8))
					}
				}
				bias := normals(r, n, 1)
				x := normals(r, m*kd, 1)
				w := newBuffer(t, d, 16+len(raw))
				copy(w.Bytes()[16:], raw)
				xb, bb, y := upload(t, d, 0, x), upload(t, d, 0, bias), newBuffer(t, d, 4*m*n)
				weights := Weights{W: w.At(16), Type: typ, N: n, K: kd}
				if biased {
					weights.Bias = bb.At(0)
				}
				run(t, d, func(c *CommandBuffer) { k.Linear(c, y.At(0), n, xb.At(0), kd, m, weights) })
				got := read[float32](y, 0, m*n)
				name := fmt.Sprintf("%s, bias %v, %d rows", typ, biased, m)
				for i := range m {
					for j := range n {
						var sum, mag float64
						for l := range kd {
							p := float64(wf[j*kd+l]) * float64(x[i*kd+l])
							sum, mag = sum+p, mag+math.Abs(p)
						}
						bound := float64(kd) * 0x1p-24 * mag
						if biased {
							sum += float64(bias[j])
							bound += math.Abs(sum) * 0x1p-24
						}
						if e := math.Abs(float64(got[i*n+j]) - sum); e > bound {
							t.Fatalf("%s: y[%d,%d] = %v, want %v (error %.3g)", name, i, j, got[i*n+j], sum, e)
						}
					}
				}
			}
		}
	}
}

// TestLayerNorm checks LayerNorm, alone and with the residual add BERT
// fuses into it, against ops (which sums in float64).
func TestLayerNorm(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(27, 28))
	const rows, dim, eps = 5, 1000, 1e-12
	for _, withResidual := range []bool{false, true} {
		x, res, w, b := normals(r, rows*dim, 3), normals(r, rows*dim, 3), normals(r, dim, 1), normals(r, dim, 1)
		xb, rb, wb, bb := upload(t, d, 0, x), upload(t, d, 0, res), upload(t, d, 0, w), upload(t, d, 0, b)
		var residual Ref
		want := tensor.FromData(slices.Clone(x), rows, dim)
		if withResidual {
			residual = rb.At(0)
			ops.AddLayerNorm(want, tensor.FromData(res, rows, dim), tensor.FromData(w, dim), tensor.FromData(b, dim), eps)
		} else {
			ops.LayerNorm(want, tensor.FromData(w, dim), tensor.FromData(b, dim), eps)
		}
		run(t, d, func(c *CommandBuffer) { k.LayerNorm(c, xb.At(0), residual, wb.At(0), bb.At(0), rows, dim, eps) })
		t.Logf("residual %v: largest error / largest value %.2g", withResidual,
			within(t, "x", read[float32](xb, 0, rows*dim), want.Data, 1e-6, 1e-6))
	}
}

// TestActivations checks GELU against ops.GELU (the same erf
// approximation), ReLU, AddRow and Scale.
func TestActivations(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(29, 30))
	const rows, cols = 7, 1543
	x, row := normals(r, rows*cols, 3), normals(r, cols, 1)
	g, rl, ar, rb := upload(t, d, 0, x), upload(t, d, 0, x), upload(t, d, 0, x), upload(t, d, 0, row)
	sc := newBuffer(t, d, 4*len(x))
	run(t, d, func(c *CommandBuffer) {
		k.GELU(c, g.At(0), rows*cols)
		k.ReLU(c, rl.At(0), rows*cols)
		k.AddRow(c, ar.At(0), rb.At(0), rows, cols)
		k.Scale(c, sc.At(0), rb.At(0), cols, 0.0441941738)
	})
	wantScale := slices.Clone(row)
	for i := range wantScale {
		wantScale[i] *= 0.0441941738
	}
	exact(t, "Scale", read[float32](sc, 0, cols), wantScale)
	wantGELU, wantReLU, wantRow := slices.Clone(x), slices.Clone(x), slices.Clone(x)
	ops.GELU(wantGELU)
	ops.ReLU(wantReLU)
	for i := range rows {
		ops.Add(wantRow[i*cols:(i+1)*cols], row)
	}
	t.Logf("GELU: largest error / largest value %.2g", within(t, "GELU", read[float32](g, 0, len(x)), wantGELU, 1e-6, 1e-7))
	exact(t, "ReLU", read[float32](rl, 0, len(x)), wantReLU)
	exact(t, "AddRow", read[float32](ar, 0, len(x)), wantRow)
}

// TestEmbedTypes looks up rows of fp16 and float32 tables.
func TestEmbedTypes(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(31, 32))
	const vocab, cols = 50, 384
	f := normals(r, vocab*cols, 0.1)
	h := make([]uint16, len(f))
	hf := make([]float32, len(f))
	for i := range h {
		h[i] = fp16(r)
		hf[i] = half.ToFloat32(h[i])
	}
	ids := []int32{49, 0, 7, 7}
	ib := upload(t, d, 0, ids)
	for _, tc := range []struct {
		typ   WeightType
		table *Buffer
		want  []float32
	}{{F32, upload(t, d, 0, f), f}, {F16, upload(t, d, 0, h), hf}} {
		out := newBuffer(t, d, 4*len(ids)*cols)
		run(t, d, func(c *CommandBuffer) { k.Embed(c, out.At(0), tc.table.At(0), tc.typ, ib.At(0), vocab, len(ids), cols) })
		var want []float32
		for _, id := range ids {
			want = append(want, tc.want[int(id)*cols:int(id+1)*cols]...)
		}
		exact(t, tc.typ.String(), read[float32](out, 0, len(want)), want)
	}
}

// TestAttendModes checks attention over every cached position (BERT's and
// T5's encoders, T5's cross-attention, where the keys are not the queries'
// positions), with and without T5's additive bias, and causal attention
// with a bias (T5's decoder), against ops.AttentionStrided, through both
// the one-row and the tiled kernel.
func TestAttendModes(t *testing.T) {
	d, k := kernels(t)
	r := rand.New(rand.NewPCG(33, 34))
	for _, tc := range []struct {
		name               string
		heads, dim         int
		rows, keys, offset int
		causal, biased     bool
	}{
		{"encoder", 12, 32, 9, 9, 0, false, false},
		{"encoder, long", 16, 64, 300, 300, 0, false, false},
		{"encoder, bias", 8, 64, 45, 45, 0, false, true},
		{"cross", 8, 64, 5, 37, 0, false, false},
		{"cross, one row", 8, 64, 1, 37, 0, false, false},
		{"decoder, bias", 8, 64, 6, 6, 0, true, true},
		{"decoder step, bias", 8, 64, 1, 20, 19, true, true},
	} {
		hidden := tc.heads * tc.dim
		q, kv := normals(r, tc.rows*hidden, 1), normals(r, tc.keys*2*hidden, 1)
		var bias []float32
		if tc.biased {
			bias = normals(r, tc.heads*tc.rows*tc.keys, 2)
		}
		const scale = 0.25
		want := make([]float32, tc.rows*hidden)
		ops.AttentionStrided(want, hidden, q, hidden, kv, 2*hidden, kv[hidden:], 2*hidden, tc.rows, tc.keys, hidden,
			ops.AttentionArgs{Heads: tc.heads, Scale: scale, Bias: bias, Causal: tc.causal, Offset: tc.offset})

		cache, err := d.NewKVCache(tc.heads, tc.dim, tc.keys)
		if err != nil {
			t.Fatal(err)
		}
		defer cache.Release()
		qb, kvb, out := upload(t, d, 0, q), upload(t, d, 0, kv), newBuffer(t, d, 4*len(want))
		args := AttendArgs{Heads: tc.heads, Scale: scale, Causal: tc.causal, Offset: tc.offset, Keys: tc.keys}
		if tc.biased {
			args.Bias = upload(t, d, 0, bias).At(0)
		}
		run(t, d, func(c *CommandBuffer) {
			k.KVAppend(c, cache, kvb.At(0), kvb.At(4*hidden), 2*hidden, tc.keys, 0)
			k.Attend(c, out.At(0), hidden, qb.At(0), hidden, cache, tc.rows, args)
		})
		t.Logf("%-18s largest error / largest value %.2g", tc.name, within(t, tc.name, read[float32](out, 0, len(want)), want, 2e-6, 2e-6))
	}
}

// fp16 returns the encoding of a random multiple of 1/256 in [-0.5, 0.5),
// which fp16 holds exactly.
func fp16(r *rand.Rand) uint16 {
	h, ok := half.FromFloat32Exact(float32(r.IntN(256)-128) / 256)
	if !ok {
		panic("not exact in fp16")
	}
	return h
}
