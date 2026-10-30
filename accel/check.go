package accel

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"unsafe"

	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// Check runs every kernel once on small inputs and compares the results
// with package ops, or with float64 for the products (bf16 and int8, one row
// and many): bit-identical where the kernel promises it (RoPE, the cache
// append, add, embedding lookup, argmax), within the bounds metal's tests
// use otherwise. It takes a few milliseconds.
func Check(d *metal.Device, k *metal.Kernels) error {
	c := &checker{d: d, seed: 1}
	defer c.release()

	// Shapes are small but not multiples of the kernels' tiles.
	const (
		n, kd, m                    = 37, 260, 3
		heads, kvHeads, dim         = 4, 2, 64
		prefix, rows, eps           = 20, 3, 1e-6
		qDim, kvDim                 = heads * dim, kvHeads * dim
		width, ff, vocab            = qDim + 2*kvDim, 70, 11
		total                       = prefix + rows
		scale               float32 = 0.125
	)
	wh, wf := c.bf16s(n * kd)
	q, qScale := ops.QuantizeInt8(tensor.FromData(wf, n, kd))
	x := c.floats(m * kd)
	normIn, normW := c.floats(rows*width), c.floats(dim)
	qkv := c.floats(total * width)
	gateUp, addA, addB := c.floats(rows*2*ff), c.floats(n), c.floats(n)
	table, tableF := c.bf16s(vocab * dim)
	ids := []int32{3, 0, 10}
	logits := c.floats(500)
	logits[123], logits[321] = 40, 40 // a tie, which goes to the first
	rot := ops.NewRoPE(dim, 1e4).Rotation(prefix, rows)
	cos, sin := rot.Tables()

	W, X := c.upload(u16(wh)), c.upload(f32(x))
	Q, QScale := c.upload(i8(q)), c.upload(f32(qScale))
	Y, Rows, YQ, RowsQ := c.buffer(4*m*n), c.buffer(4*m*n), c.buffer(4*m*n), c.buffer(4*m*n)
	Norm, NormW := c.upload(f32(normIn)), c.upload(f32(normW))
	Rope, Cos, Sin := c.upload(f32(qkv[prefix*width:])), c.upload(f32(cos)), c.upload(f32(sin))
	QKV, Ctx := c.upload(f32(qkv)), c.buffer(4*rows*qDim)
	GateUp, Act := c.upload(f32(gateUp)), c.buffer(4*rows*ff)
	AddA, AddB := c.upload(f32(addA)), c.upload(f32(addB))
	Table, IDs, Embedded := c.upload(u16(table)), c.upload(i32(ids)), c.buffer(4*len(ids)*dim)
	Logits, Best := c.upload(f32(logits)), c.buffer(4)
	kv, err := d.NewKVCache(kvHeads, dim, total)
	if err != nil || c.err != nil {
		return errors.Join(err, c.err)
	}
	defer kv.Release()

	cb := d.NewCommandBuffer()
	bf16 := metal.Weights{W: W.At(0), Type: metal.BF16, N: n, K: kd}
	int8 := metal.Weights{W: Q.At(0), Scale: QScale.At(0), Type: metal.Int8, N: n, K: kd}
	k.Linear(cb, Y.At(0), n, X.At(0), kd, m, bf16)
	k.Linear(cb, YQ.At(0), n, X.At(0), kd, m, int8)
	for i := range m {
		k.Linear(cb, Rows.At(4*i*n), n, X.At(4*i*kd), kd, 1, bf16)
		k.Linear(cb, RowsQ.At(4*i*n), n, X.At(4*i*kd), kd, 1, int8)
	}
	k.RMSNorm(cb, Norm.At(0), Norm.At(0), NormW.At(0), width, rows, heads, dim, eps)
	k.RoPE(cb, Rope.At(0), Cos.At(0), Sin.At(0), width, rows, heads+kvHeads, dim)
	k.KVAppend(cb, kv, QKV.At(4*qDim), QKV.At(4*(qDim+kvDim)), width, total, 0)
	k.Attend(cb, Ctx.At(0), qDim, QKV.At(4*prefix*width), width, kv, rows, metal.AttendArgs{Heads: heads, Scale: scale, Causal: true, Offset: prefix})
	k.SwiGLU(cb, Act.At(0), GateUp.At(0), rows, ff)
	k.Add(cb, AddA.At(0), AddB.At(0), n)
	k.Embed(cb, Embedded.At(0), Table.At(0), metal.BF16, IDs.At(0), vocab, len(ids), dim)
	k.Argmax(cb, Best.At(0), Logits.At(0), len(logits))
	if err := cb.Wait(); err != nil {
		return err
	}

	var errs []error
	fail := func(format string, args ...any) { errs = append(errs, fmt.Errorf(format, args...)) }

	// Each product output must be within float32 rounding of the float64
	// sum (and, for int8, of the scaling).
	for _, p := range []struct {
		name string
		y    *metal.Buffer
		w    func(j, l int) float64
		s    func(j int) float64
	}{
		{"Linear bf16, many rows", Y, func(j, l int) float64 { return float64(wf[j*kd+l]) }, nil},
		{"Linear bf16, one row", Rows, func(j, l int) float64 { return float64(wf[j*kd+l]) }, nil},
		{"Linear int8, many rows", YQ, func(j, l int) float64 { return float64(q[j*kd+l]) }, func(j int) float64 { return float64(qScale[j]) }},
		{"Linear int8, one row", RowsQ, func(j, l int) float64 { return float64(q[j*kd+l]) }, func(j int) float64 { return float64(qScale[j]) }},
	} {
		got := floats(p.y, m*n)
	outputs:
		for i := range m {
			for j := range n {
				var sum, mag float64
				for l := range kd {
					v := p.w(j, l) * float64(x[i*kd+l])
					sum, mag = sum+v, mag+math.Abs(v)
				}
				bound := float64(kd) * 0x1p-24 * mag
				if p.s != nil {
					sum, bound = sum*p.s(j), (bound+math.Abs(sum)*0x1p-24)*p.s(j)
				}
				if math.Abs(float64(got[i*n+j])-sum) > bound {
					fail("%s row %d output %d = %v, want %v", p.name, i, j, got[i*n+j], sum)
					break outputs
				}
			}
		}
	}

	wantNorm := slices.Clone(normIn)
	for r := range rows {
		ops.RMSNorm(tensor.FromData(wantNorm[r*width:r*width+qDim], heads, dim), tensor.FromData(normW, dim), eps)
	}
	near(fail, "RMSNorm", floats(Norm, len(wantNorm)), wantNorm, 1e-6, 1e-7)

	wantRope := slices.Clone(qkv[prefix*width:])
	rot.Apply(wantRope, width, heads)
	rot.Apply(wantRope[qDim:], width, kvHeads)
	if !slices.Equal(floats(Rope, len(wantRope)), wantRope) {
		fail("RoPE differs from ops.Rotation.Apply")
	}

	cache := ops.NewKVCache(kvHeads, dim, total)
	cache.Append(qkv[qDim:], width, qkv[qDim+kvDim:], width, total)
	wantCtx := make([]float32, rows*qDim)
	cache.Attend(wantCtx, qDim, qkv[prefix*width:], width, rows, ops.AttentionArgs{
		Heads: heads, KVHeads: kvHeads, Scale: scale, Causal: true, Offset: prefix,
	})
	near(fail, "Attend", floats(Ctx, len(wantCtx)), wantCtx, 2e-6, 2e-6)
	keys := floats(kv.Keys, kvHeads*total*dim)
	if !slices.Equal(keys[(total-1)*dim:total*dim], qkv[(total-1)*width+qDim:][:dim]) {
		fail("KVAppend wrote the wrong keys")
	}

	wantAct := tensor.New(rows, ff)
	ops.SwiGLUInto(wantAct, tensor.FromData(gateUp, rows, 2*ff))
	near(fail, "SwiGLU", floats(Act, rows*ff), wantAct.Data, 1e-6, 1e-12)

	wantAdd := slices.Clone(addA)
	ops.Add(wantAdd, addB)
	if !slices.Equal(floats(AddA, n), wantAdd) {
		fail("Add differs from ops.Add")
	}
	var wantEmbed []float32
	for _, id := range ids {
		wantEmbed = append(wantEmbed, tableF[int(id)*dim:int(id+1)*dim]...)
	}
	if !slices.Equal(floats(Embedded, len(wantEmbed)), wantEmbed) {
		fail("Embed differs from the table")
	}
	if got, want := int(metal.Slice[int32](Best)[0]), ops.Argmax(logits); got != want {
		fail("Argmax = %d, want %d", got, want)
	}
	return errors.Join(errs...)
}

// near reports values outside |got-want| ≤ rel·|want| + abs.
func near(fail func(string, ...any), name string, got, want []float32, rel, abs float64) {
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); !(d <= rel*math.Abs(float64(want[i]))+abs) {
			fail("%s[%d] = %v, want %v", name, i, got[i], want[i])
			return
		}
	}
}

// checker makes deterministic inputs and the buffers holding them.
type checker struct {
	d       *metal.Device
	seed    uint64
	buffers []*metal.Buffer
	err     error
}

// next returns a pseudo-random value in [-1, 1).
func (c *checker) next() float32 {
	c.seed = c.seed*6364136223846793005 + 1442695040888963407
	return float32(int32(c.seed>>32)) / (1 << 31)
}

func (c *checker) floats(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = c.next()
	}
	return v
}

// bf16s returns n bf16 values and their float32 values.
func (c *checker) bf16s(n int) ([]uint16, []float32) {
	h, f := make([]uint16, n), make([]float32, n)
	for i := range h {
		h[i] = uint16(math.Float32bits(c.next()) >> 16)
		f[i] = math.Float32frombits(uint32(h[i]) << 16)
	}
	return h, f
}

func (c *checker) buffer(n int) *metal.Buffer {
	b, err := c.d.NewBuffer(n)
	if err != nil {
		c.err = err
		b, _ = c.d.NewBuffer(16) // keeps the caller simple; Check returns c.err before using it
	}
	c.buffers = append(c.buffers, b)
	return b
}

func (c *checker) upload(data []byte) *metal.Buffer {
	b := c.buffer(len(data))
	copy(b.Bytes(), data)
	return b
}

func (c *checker) release() {
	for _, b := range c.buffers {
		b.Release()
	}
}

func floats(b *metal.Buffer, n int) []float32 { return slices.Clone(metal.Slice[float32](b)[:n]) }

func f32(v []float32) []byte { return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)) }
func u16(v []uint16) []byte  { return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 2*len(v)) }
func i32(v []int32) []byte   { return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 4*len(v)) }
func i8(v []int8) []byte     { return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)) }
