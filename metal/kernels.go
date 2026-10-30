package metal

import (
	"bytes"
	"cmp"
	"embed"
	"fmt"
	"io/fs"
)

//go:embed kernels/*.metal
var kernelFiles embed.FS

// kernelSource is common.metal, then every other file in kernels/, as one
// compilation unit.
func kernelSource() string {
	const common = "kernels/common.metal"
	src, _ := kernelFiles.ReadFile(common)
	b := bytes.NewBuffer(src)
	names, _ := fs.Glob(kernelFiles, "kernels/*.metal")
	for _, name := range names {
		if name != common {
			src, _ := kernelFiles.ReadFile(name)
			b.WriteByte('\n')
			b.Write(src)
		}
	}
	return b.String()
}

// Ref is where a kernel operand starts: a byte offset into a buffer.
type Ref struct {
	Buf *Buffer
	Off int
}

// At returns a Ref to byte offset off of b.
func (b *Buffer) At(off int) Ref { return Ref{b, off} }

// SetRef binds r to the kernel's [[buffer(index)]].
func (c *CommandBuffer) SetRef(index int, r Ref) { c.SetBuffer(index, r.Buf, r.Off) }

// need panics unless r has room for n bytes and is aligned to align bytes.
func need(op, operand string, r Ref, n, align int) {
	if r.Buf == nil || r.Off < 0 || r.Off+n > r.Buf.Len() || r.Off%align != 0 {
		size := 0
		if r.Buf != nil {
			size = r.Buf.Len()
		}
		panic(fmt.Sprintf("metal.Kernels.%s: %s needs %d bytes at offset %d, aligned to %d, in a %d-byte buffer",
			op, operand, n, r.Off, align, size))
	}
}

// Kernels are the inference kernels in kernels/, compiled for a device. Each
// method records one dispatch into a CommandBuffer; operands are float32
// unless named otherwise, and shapes are checked before Metal sees them.
// Kernels is safe for concurrent use.
type Kernels struct {
	gemv, gemm, embed                                          map[WeightType]*Pipeline
	rmsnorm, layernorm, swiglu, gelu, relu, add, addRow, scale *Pipeline
	argmax, rope, kvAppend, attend, attendTiled                *Pipeline
	all                                                        []*Pipeline
}

// WeightType is how a linear layer's weights (or an embedding table) are
// stored on the GPU.
type WeightType int

const (
	BF16 WeightType = iota
	F16
	F32
	Int8 // with a float32 scale per output, as ops.PackInt8
)

func (t WeightType) String() string { return [...]string{"bf16", "f16", "f32", "int8"}[t] }

// Size returns the bytes per value.
func (t WeightType) Size() int { return [...]int{2, 2, 4, 1}[t] }

// The gemv instantiation used for every weight type: rows per simdgroup and
// simdgroups per threadgroup, chosen with BenchmarkGemvVariants.
const (
	gemvRows   = 4
	gemvGroups = 8
)

// mmaTile is the gemm_mma instantiation used for every weight type, chosen
// with BenchmarkGemmVariants: a threadgroup's block of y and step through k,
// and its grid of simdgroups.
var mmaTile = mmaShape{64, 64, 16, 2, 4}

type mmaShape struct{ bm, bn, bk, wm, wn int }

func (s mmaShape) name(weights string) string {
	return fmt.Sprintf("gemm_mma_%s_%dx%dx%d_%dx%d", weights, s.bm, s.bn, s.bk, s.wm, s.wn)
}

// Kernels compiles the kernels.
func (d *Device) Kernels() (*Kernels, error) {
	lib, err := d.Compile(kernelSource())
	if err != nil {
		return nil, err
	}
	defer lib.Release()
	k := &Kernels{gemv: map[WeightType]*Pipeline{}, gemm: map[WeightType]*Pipeline{}, embed: map[WeightType]*Pipeline{}}
	load := func(name string) *Pipeline {
		if err != nil {
			return nil
		}
		var p *Pipeline
		if p, err = lib.Pipeline(name); err == nil {
			k.all = append(k.all, p)
		}
		return p
	}
	k.rmsnorm, k.layernorm, k.swiglu = load("rmsnorm"), load("layernorm"), load("swiglu")
	k.gelu, k.relu, k.add, k.addRow, k.scale = load("gelu"), load("relu"), load("add"), load("add_row"), load("scale")
	k.argmax, k.rope, k.kvAppend = load("argmax"), load("rope"), load("kv_append")
	k.attend, k.attendTiled = load("attend"), load("attend_tiled")
	for _, t := range []WeightType{BF16, F16, F32, Int8} {
		k.gemv[t] = load(fmt.Sprintf("gemv_%s_r%d_sg%d", t, gemvRows, gemvGroups))
		k.gemm[t] = load(mmaTile.name(t.String()))
		if t != Int8 {
			k.embed[t] = load("embed_" + t.String())
		}
	}
	if err != nil {
		k.Release()
		return nil, err
	}
	return k, nil
}

// Release releases the pipelines.
func (k *Kernels) Release() {
	for _, p := range k.all {
		p.Release()
	}
	k.all = nil
}

// Weights are a linear layer's weights on the GPU: W [N, K] row-major (K a
// multiple of 4, aligned to 4 values) in Type, with a float32 Scale [N] for
// Int8 (ops.QuantizeInt8 gives W and Scale in this layout) and an optional
// float32 Bias [N].
type Weights struct {
	W, Scale, Bias Ref
	Type           WeightType
	N, K           int
}

// Linear records y = x·Wᵀ + bias for m rows of x (row i at float i·ldx, ldx
// a multiple of 4) into m rows of y (row i at float i·ldy). One row runs a
// matrix-vector kernel that streams the weights; more run a kernel on the
// simdgroup matrix units, much faster for many rows but summing in another
// order, so a row's outputs are not bit-identical to the one-row kernel's.
func (k *Kernels) Linear(c *CommandBuffer, y Ref, ldy int, x Ref, ldx, m int, w Weights) {
	n, kdim := w.N, w.K
	if m <= 0 || n <= 0 || kdim <= 0 || kdim%4 != 0 || ldx < kdim || ldx%4 != 0 || ldy < n {
		panic(fmt.Sprintf("metal.Kernels.Linear: %d rows by a %d×%d %s weight, ldx %d, ldy %d", m, n, kdim, w.Type, ldx, ldy))
	}
	need("Linear", "y", y, 4*((m-1)*ldy+n), 4)
	need("Linear", "w", w.W, w.Type.Size()*n*kdim, 4*w.Type.Size())
	need("Linear", "x", x, 4*((m-1)*ldx+kdim), 16)
	if w.Type == Int8 {
		need("Linear", "scale", w.Scale, 4*n, 4)
	}
	if w.Bias.Buf != nil {
		need("Linear", "bias", w.Bias, 4*n, 4)
	}
	if m == 1 {
		gemv(c, k.gemv[w.Type], gemvRows, gemvGroups, y, x, w)
		return
	}
	gemmMMA(c, k.gemm[w.Type], mmaTile, y, ldy, x, ldx, m, w)
}

// productArgs binds a product's weights, scale, bias and arguments; any
// buffer stands in for a scale or bias the kernel will not read.
func productArgs(c *CommandBuffer, y, x Ref, m, ldx, ldy int, w Weights) {
	c.SetRef(0, w.W)
	c.SetRef(1, x)
	c.SetRef(2, y)
	var bias uint32
	if w.Bias.Buf != nil {
		bias = 1
	}
	SetValue(c, 3, [6]uint32{uint32(w.N), uint32(w.K), uint32(m), uint32(ldx), uint32(ldy), bias})
	c.SetRef(4, cmp.Or(w.Scale, w.W))
	c.SetRef(5, cmp.Or(w.Bias, w.W))
}

// gemv records a gemv instantiation with rows rows per simdgroup and groups
// simdgroups per threadgroup.
func gemv(c *CommandBuffer, p *Pipeline, rows, groups int, y, x Ref, w Weights) {
	c.SetPipeline(p)
	productArgs(c, y, x, 1, w.K, w.N, w)
	perGroup := rows * groups
	c.DispatchThreadgroups(Size{(w.N + perGroup - 1) / perGroup, 1, 1}, Size{32 * groups, 1, 1})
}

// gemmMMA records a gemm_mma instantiation.
func gemmMMA(c *CommandBuffer, p *Pipeline, s mmaShape, y Ref, ldy int, x Ref, ldx, m int, w Weights) {
	c.SetPipeline(p)
	productArgs(c, y, x, m, ldx, ldy, w)
	c.DispatchThreadgroups(Size{(w.N + s.bn - 1) / s.bn, (m + s.bm - 1) / s.bm, 1}, Size{32 * s.wm * s.wn, 1, 1})
}

// RMSNorm records dst = RMSNorm(src)·w for rows rows of groups groups of dim
// values: group g of row r starts at float r·ld + g·dim in src and dst, which
// may be the same. w holds dim weights. groups is 1 for a layer's norm and
// the number of heads for Qwen3's per-head query and key norms.
func (k *Kernels) RMSNorm(c *CommandBuffer, dst, src, w Ref, ld, rows, groups, dim int, eps float32) {
	if rows <= 0 || groups <= 0 || dim <= 0 || ld < groups*dim {
		panic(fmt.Sprintf("metal.Kernels.RMSNorm: %d rows of %d groups of %d, ld %d", rows, groups, dim, ld))
	}
	span := 4 * ((rows-1)*ld + groups*dim)
	need("RMSNorm", "src", src, span, 4)
	need("RMSNorm", "dst", dst, span, 4)
	need("RMSNorm", "w", w, 4*dim, 4)
	c.SetPipeline(k.rmsnorm)
	c.SetRef(0, src)
	c.SetRef(1, dst)
	c.SetRef(2, w)
	SetValue(c, 3, struct {
		ld, groups, dim uint32
		eps             float32
	}{uint32(ld), uint32(groups), uint32(dim), eps})
	c.DispatchThreadgroups(Size{rows * groups, 1, 1}, Size{min(1024, max(32, (dim+31)/32*32)), 1, 1})
}

// SwiGLU records out [rows, ff] = SiLU(gate)·up, where row r of gateUp
// [rows, 2·ff] holds gate then up.
func (k *Kernels) SwiGLU(c *CommandBuffer, out, gateUp Ref, rows, ff int) {
	if rows <= 0 || ff <= 0 {
		panic(fmt.Sprintf("metal.Kernels.SwiGLU: %d rows of %d", rows, ff))
	}
	need("SwiGLU", "out", out, 4*rows*ff, 4)
	need("SwiGLU", "gateUp", gateUp, 8*rows*ff, 4)
	c.SetPipeline(k.swiglu)
	c.SetRef(0, gateUp)
	c.SetRef(1, out)
	SetValue(c, 2, uint32(ff))
	c.DispatchThreads(Size{ff, rows, 1}, Size{256, 1, 1})
}

// Add records dst += src over n values.
func (k *Kernels) Add(c *CommandBuffer, dst, src Ref, n int) {
	if n <= 0 {
		panic(fmt.Sprintf("metal.Kernels.Add: %d values", n))
	}
	need("Add", "dst", dst, 4*n, 4)
	need("Add", "src", src, 4*n, 4)
	c.SetPipeline(k.add)
	c.SetRef(0, dst)
	c.SetRef(1, src)
	c.DispatchThreads(Size{n, 1, 1}, Size{256, 1, 1})
}

// Embed records dst [rows, cols] = the rows of a table [vocab, cols] of
// bf16, fp16 or float32 values chosen by ids, rows int32 token ids, which
// the caller has checked against vocab.
func (k *Kernels) Embed(c *CommandBuffer, dst, table Ref, t WeightType, ids Ref, vocab, rows, cols int) {
	if rows <= 0 || cols <= 0 || vocab <= 0 || t == Int8 {
		panic(fmt.Sprintf("metal.Kernels.Embed: %d rows of %d from %d %s", rows, cols, vocab, t))
	}
	need("Embed", "dst", dst, 4*rows*cols, 4)
	need("Embed", "table", table, t.Size()*vocab*cols, t.Size())
	need("Embed", "ids", ids, 4*rows, 4)
	c.SetPipeline(k.embed[t])
	c.SetRef(0, table)
	c.SetRef(1, ids)
	c.SetRef(2, dst)
	SetValue(c, 3, uint32(cols))
	c.DispatchThreads(Size{cols, rows, 1}, Size{256, 1, 1})
}

// Scale records dst = s·src over n values; dst may be src.
func (k *Kernels) Scale(c *CommandBuffer, dst, src Ref, n int, s float32) {
	if n <= 0 {
		panic(fmt.Sprintf("metal.Kernels.Scale: %d values", n))
	}
	need("Scale", "dst", dst, 4*n, 4)
	need("Scale", "src", src, 4*n, 4)
	c.SetPipeline(k.scale)
	c.SetRef(0, src)
	c.SetRef(1, dst)
	SetValue(c, 2, s)
	c.DispatchThreads(Size{n, 1, 1}, Size{256, 1, 1})
}

// AddRow records adding row [cols] to every row of x [rows, cols].
func (k *Kernels) AddRow(c *CommandBuffer, x, row Ref, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("metal.Kernels.AddRow: %d rows of %d", rows, cols))
	}
	need("AddRow", "x", x, 4*rows*cols, 4)
	need("AddRow", "row", row, 4*cols, 4)
	c.SetPipeline(k.addRow)
	c.SetRef(0, x)
	c.SetRef(1, row)
	SetValue(c, 2, uint32(cols))
	c.DispatchThreads(Size{cols, rows, 1}, Size{256, 1, 1})
}

// LayerNorm records x = LayerNorm(x + residual)·w + b for each row of x
// [rows, dim], in place; residual is optional (Buf nil).
func (k *Kernels) LayerNorm(c *CommandBuffer, x, residual, w, b Ref, rows, dim int, eps float32) {
	if rows <= 0 || dim <= 0 {
		panic(fmt.Sprintf("metal.Kernels.LayerNorm: %d rows of %d", rows, dim))
	}
	need("LayerNorm", "x", x, 4*rows*dim, 4)
	need("LayerNorm", "w", w, 4*dim, 4)
	need("LayerNorm", "b", b, 4*dim, 4)
	var add uint32
	if residual.Buf != nil {
		need("LayerNorm", "residual", residual, 4*rows*dim, 4)
		add = 1
	}
	c.SetPipeline(k.layernorm)
	c.SetRef(0, x)
	c.SetRef(1, cmp.Or(residual, x))
	c.SetRef(2, w)
	c.SetRef(3, b)
	SetValue(c, 4, struct {
		dim, residual uint32
		eps           float32
	}{uint32(dim), add, eps})
	c.DispatchThreadgroups(Size{rows, 1, 1}, Size{min(1024, max(32, (dim+31)/32*32)), 1, 1})
}

// GELU records the exact (erf) GELU over n values of x, in place, as
// ops.GELU computes it.
func (k *Kernels) GELU(c *CommandBuffer, x Ref, n int) { k.unary(c, "GELU", k.gelu, x, n) }

// ReLU records replacing negative values among n values of x with zero.
func (k *Kernels) ReLU(c *CommandBuffer, x Ref, n int) { k.unary(c, "ReLU", k.relu, x, n) }

func (k *Kernels) unary(c *CommandBuffer, op string, p *Pipeline, x Ref, n int) {
	if n <= 0 {
		panic(fmt.Sprintf("metal.Kernels.%s: %d values", op, n))
	}
	need(op, "x", x, 4*n, 4)
	c.SetPipeline(p)
	c.SetRef(0, x)
	c.DispatchThreads(Size{n, 1, 1}, Size{256, 1, 1})
}

// Argmax records out[0] (int32) = the index of the largest of x[0..n), the
// first one on ties.
func (k *Kernels) Argmax(c *CommandBuffer, out, x Ref, n int) {
	if n <= 0 {
		panic(fmt.Sprintf("metal.Kernels.Argmax: %d values", n))
	}
	need("Argmax", "out", out, 4, 4)
	need("Argmax", "x", x, 4*n, 4)
	c.SetPipeline(k.argmax)
	c.SetRef(0, x)
	c.SetRef(1, out)
	SetValue(c, 2, uint32(n))
	c.DispatchThreadgroups(Size{1, 1, 1}, Size{1024, 1, 1})
}

// RoPE records the rotation of heads consecutive heads of dim values at the
// start of each of rows rows of x (row r at float r·ld), in place, by the
// angles whose cosines and sines are cos and sin [rows, dim/2]
// (ops.Rotation.Tables). A query's heads and its key's heads, adjacent in a
// stacked q|k|v row, rotate in one call.
func (k *Kernels) RoPE(c *CommandBuffer, x, cos, sin Ref, ld, rows, heads, dim int) {
	if rows <= 0 || heads <= 0 || dim <= 0 || dim%2 != 0 || ld < heads*dim {
		panic(fmt.Sprintf("metal.Kernels.RoPE: %d rows of %d heads of %d, ld %d", rows, heads, dim, ld))
	}
	half := dim / 2
	need("RoPE", "x", x, 4*((rows-1)*ld+heads*dim), 4)
	need("RoPE", "cos", cos, 4*rows*half, 4)
	need("RoPE", "sin", sin, 4*rows*half, 4)
	c.SetPipeline(k.rope)
	c.SetRef(0, x)
	c.SetRef(1, cos)
	c.SetRef(2, sin)
	SetValue(c, 3, [2]uint32{uint32(ld), uint32(half)})
	c.DispatchThreads(Size{half, heads, rows}, Size{min(half, 256), 1, 1})
}

// KVCache is one layer's attention keys and values on the GPU: two buffers
// of [Heads, Capacity, Dim] float32, a head's positions contiguous.
type KVCache struct {
	Keys, Values         *Buffer
	Heads, Dim, Capacity int
}

// NewKVCache allocates a cache for heads heads of dim values and capacity
// positions.
func (d *Device) NewKVCache(heads, dim, capacity int) (*KVCache, error) {
	if heads <= 0 || dim <= 0 || capacity <= 0 {
		panic(fmt.Sprintf("metal.NewKVCache: %d heads of %d, %d positions", heads, dim, capacity))
	}
	n := 4 * heads * dim * capacity
	keys, err := d.NewBuffer(n)
	if err != nil {
		return nil, err
	}
	values, err := d.NewBuffer(n)
	if err != nil {
		keys.Release()
		return nil, err
	}
	return &KVCache{Keys: keys, Values: values, Heads: heads, Dim: dim, Capacity: capacity}, nil
}

// Release releases the cache's buffers.
func (kv *KVCache) Release() {
	kv.Keys.Release()
	kv.Values.Release()
}

// KVAppend records copying rows rows of keys and values into the cache at
// positions pos..pos+rows-1. Row r of each starts at float r·ld of key and
// value and holds the cache's heads·dim values.
func (k *Kernels) KVAppend(c *CommandBuffer, kv *KVCache, key, value Ref, ld, rows, pos int) {
	if rows <= 0 || pos < 0 || pos+rows > kv.Capacity || ld < kv.Heads*kv.Dim {
		panic(fmt.Sprintf("metal.Kernels.KVAppend: %d rows at %d into %d positions, ld %d", rows, pos, kv.Capacity, ld))
	}
	span := 4 * ((rows-1)*ld + kv.Heads*kv.Dim)
	need("KVAppend", "key", key, span, 4)
	need("KVAppend", "value", value, span, 4)
	c.SetPipeline(k.kvAppend)
	c.SetRef(0, key)
	c.SetRef(1, value)
	c.SetBuffer(2, kv.Keys, 0)
	c.SetBuffer(3, kv.Values, 0)
	SetValue(c, 4, [4]uint32{uint32(ld), uint32(kv.Dim), uint32(pos), uint32(kv.Capacity)})
	c.DispatchThreads(Size{kv.Dim, kv.Heads, rows}, Size{min(kv.Dim, 256), 1, 1})
}

// AttendArgs describe one attention call.
type AttendArgs struct {
	Heads int // query heads; head h reads key/value head h/(Heads/kv.Heads)
	Scale float32
	// Causal: row r is position Offset+r and sees positions 0..Offset+r, all
	// of which must be in the cache. Otherwise every row sees positions
	// 0..Keys-1.
	Causal bool
	Offset int
	Keys   int
	// Bias, if set, is added to the scaled scores: [Heads, rows, keys]
	// float32, keys being Offset+rows when Causal and Keys otherwise (T5's
	// relative position bias, laid out as for ops.AttentionArgs).
	Bias Ref
}

// Attend records attention for rows query rows (row r at float r·ldq of q,
// a.Heads·kv.Dim wide) over the cache, writing row r's context to float
// r·ldo of out.
//
// One row runs the decoding kernel; more run attend_tiled on the
// simdgroup matrix units (for heads of up to 128 values), which computes
// the same attention but sums in another order.
func (k *Kernels) Attend(c *CommandBuffer, out Ref, ldo int, q Ref, ldq int, kv *KVCache, rows int, a AttendArgs) {
	dim, heads, keys := kv.Dim, a.Heads, a.Keys
	if a.Causal {
		keys = a.Offset + rows
	}
	if rows <= 0 || heads <= 0 || heads%kv.Heads != 0 || dim%32 != 0 || dim > 256 || a.Offset < 0 ||
		keys <= 0 || keys > kv.Capacity || ldq < heads*dim || ldo < heads*dim {
		panic(fmt.Sprintf("metal.Kernels.Attend: %d rows of %d heads over %d heads of %d, %d positions of %d, ldq %d, ldo %d",
			rows, heads, kv.Heads, dim, keys, kv.Capacity, ldq, ldo))
	}
	need("Attend", "q", q, 4*((rows-1)*ldq+heads*dim), 4)
	need("Attend", "out", out, 4*((rows-1)*ldo+heads*dim), 4)
	var causal, bias uint32
	if a.Causal {
		causal = 1
	}
	if a.Bias.Buf != nil {
		need("Attend", "bias", a.Bias, 4*heads*rows*keys, 4)
		bias = 1
	}
	tiled := rows > 1 && dim <= 128 && dim%8 == 0
	if tiled {
		c.SetPipeline(k.attendTiled)
	} else {
		c.SetPipeline(k.attend)
	}
	c.SetRef(0, q)
	c.SetBuffer(1, kv.Keys, 0)
	c.SetBuffer(2, kv.Values, 0)
	c.SetRef(3, out)
	SetValue(c, 4, struct {
		ldq, ldo, heads, kvHeads, dim, capacity, offset, rows, keys, causal, bias uint32
		scale                                                                     float32
	}{uint32(ldq), uint32(ldo), uint32(heads), uint32(kv.Heads), uint32(dim), uint32(kv.Capacity),
		uint32(a.Offset), uint32(rows), uint32(keys), causal, bias, a.Scale})
	c.SetRef(5, cmp.Or(a.Bias, out))
	if tiled {
		c.DispatchThreadgroups(Size{heads, (rows + 31) / 32, 1}, Size{128, 1, 1})
		return
	}
	c.DispatchThreadgroups(Size{heads, rows, 1}, Size{32 * 8, 1, 1})
}
