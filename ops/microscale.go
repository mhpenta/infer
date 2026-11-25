package ops

import (
	"fmt"
	"math"
	"runtime"
)

// Microscaled weights are matrices stored as small floats with a scale per
// short block of each row, the NVFP4 and MXFP8 formats of NVIDIA's ModelOpt
// and the OCP microscaling spec. They stay in the checkpoint's bytes (often
// a view into the memory-mapped file), and products decode them as they go.
//
//   - NVFP4: two 4-bit E2M1 codes per byte, the even element in the low
//     nibble; one E4M3 scale per 16 values; and one float32 factor for the
//     whole matrix. A weight is e2m1 · scale · factor.
//   - MXFP8: one E4M3 value per byte, and one E8M0 scale, 2^(s−127), per 32
//     values. A weight is e4m3 · 2^(s−127).
type Microscaled struct {
	format     MicroFormat
	rows, cols int
	values     []byte
	scales     []byte
	factor     float32
}

// MicroFormat is the storage of a Microscaled matrix.
type MicroFormat uint8

const (
	NVFP4 MicroFormat = iota + 1
	MXFP8
)

func (f MicroFormat) String() string {
	switch f {
	case NVFP4:
		return "NVFP4"
	case MXFP8:
		return "MXFP8"
	}
	return fmt.Sprintf("MicroFormat(%d)", uint8(f))
}

// BlockSize is how many consecutive values of a row share one scale.
func (f MicroFormat) BlockSize() int {
	if f == NVFP4 {
		return 16
	}
	return 32
}

// NewNVFP4 wraps an NVFP4 matrix [rows, cols]: codes holds rows·cols/2
// bytes, scales rows·cols/16 E4M3 bytes, and factor multiplies every weight
// (ModelOpt's amax / outer_scale_denominator). The slices are not copied.
func NewNVFP4(rows, cols int, codes, scales []byte, factor float32) (*Microscaled, error) {
	if rows <= 0 || cols <= 0 || cols%16 != 0 {
		return nil, fmt.Errorf("ops: NVFP4 matrix [%d, %d] needs whole blocks of 16 per row", rows, cols)
	}
	if len(codes) != rows*cols/2 || len(scales) != rows*cols/16 {
		return nil, fmt.Errorf("ops: NVFP4 matrix [%d, %d] has %d code and %d scale bytes, want %d and %d",
			rows, cols, len(codes), len(scales), rows*cols/2, rows*cols/16)
	}
	return &Microscaled{format: NVFP4, rows: rows, cols: cols, values: codes, scales: scales, factor: factor}, nil
}

// NewMXFP8 wraps an MXFP8 matrix [rows, cols]: values holds rows·cols E4M3
// bytes and scales rows·cols/32 E8M0 exponents. The slices are not copied.
func NewMXFP8(rows, cols int, values, scales []byte) (*Microscaled, error) {
	if rows <= 0 || cols <= 0 || cols%32 != 0 {
		return nil, fmt.Errorf("ops: MXFP8 matrix [%d, %d] needs whole blocks of 32 per row", rows, cols)
	}
	if len(values) != rows*cols || len(scales) != rows*cols/32 {
		return nil, fmt.Errorf("ops: MXFP8 matrix [%d, %d] has %d value and %d scale bytes, want %d and %d",
			rows, cols, len(values), len(scales), rows*cols, rows*cols/32)
	}
	return &Microscaled{format: MXFP8, rows: rows, cols: cols, values: values, scales: scales, factor: 1}, nil
}

// Format returns the storage format.
func (w *Microscaled) Format() MicroFormat { return w.format }

// Rows returns the number of rows (a linear layer's outputs).
func (w *Microscaled) Rows() int { return w.rows }

// Cols returns the number of columns (a linear layer's inputs).
func (w *Microscaled) Cols() int { return w.cols }

// e2m1 is the value of each 4-bit E2M1 code: sign, two exponent bits, one
// mantissa bit.
var e2m1 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// e4m3 is the value of each E4M3 (float8_e4m3fn) byte: bias 7, subnormals,
// no infinities, and NaN at S.1111.111.
var e4m3 = func() (t [256]float32) {
	for b := range 256 {
		sign := float32(1)
		if b&0x80 != 0 {
			sign = -1
		}
		exp, mant := (b>>3)&0xF, b&7
		switch {
		case exp == 0xF && mant == 7:
			t[b] = float32(math.NaN())
		case exp == 0:
			t[b] = sign * float32(mant) / 8 * float32(math.Ldexp(1, -6))
		default:
			t[b] = sign * (1 + float32(mant)/8) * float32(math.Ldexp(1, exp-7))
		}
	}
	return t
}()

// E4M3 returns the value of an E4M3 (float8_e4m3fn) byte.
func E4M3(b byte) float32 { return e4m3[b] }

// e8m0Values is each MXFP8 scale byte's value, 2^(s−127), and NaN for 255.
var e8m0Values = func() (t [256]float32) {
	for s := range 255 {
		t[s] = float32(math.Ldexp(1, s-127))
	}
	t[255] = float32(math.NaN())
	return t
}()

// e8m0 returns the MXFP8 scale 2^(s−127), and NaN for s = 255.
func e8m0(s byte) float32 { return e8m0Values[s] }

// blockScale returns the scale of block b of the matrix (row-major blocks).
func (w *Microscaled) blockScale(b int) float32 {
	if w.format == NVFP4 {
		return e4m3[w.scales[b]]
	}
	return e8m0(w.scales[b])
}

// DequantizeRow writes row r as float32 to dst[:Cols()], each weight
// rounded once from its exact block value, as ModelOpt's dequantization
// (and transformers' Inkling2LoopDequantizeExperts) computes it: for NVFP4,
// (e2m1 · scale) · factor; for MXFP8, e4m3 · 2^(s−127).
func (w *Microscaled) DequantizeRow(dst []float32, r int) {
	bs := w.format.BlockSize()
	blocks := w.cols / bs
	dst = dst[:w.cols]
	for b := range blocks {
		s := w.blockScale(r*blocks + b)
		out := dst[b*bs : (b+1)*bs]
		if w.format == NVFP4 {
			codes := w.values[(r*w.cols+b*bs)/2:][:bs/2]
			for i, c := range codes {
				out[2*i] = e2m1[c&0xF] * s * w.factor
				out[2*i+1] = e2m1[c>>4] * s * w.factor
			}
			continue
		}
		for i, v := range w.values[r*w.cols+b*bs:][:bs] {
			out[i] = e4m3[v] * s
		}
	}
}

// Dequantize writes rows [r0, r0+len(dst)/Cols()) as float32 to dst.
func (w *Microscaled) Dequantize(dst []float32, r0 int) {
	n := len(dst) / w.cols
	forEach(n, workersFor(n*w.cols/4), func(i int) {
		w.DequantizeRow(dst[i*w.cols:(i+1)*w.cols], r0+i)
	})
}

// microRowsPerTask is how many rows one task of MicroLinearInto computes.
const microRowsPerTask = 32

// MicroLinearInto sets y[i·ldy + j] to the product of row i of x (x[i·ldx:],
// Cols() values) with row j of w, for n rows of x and every row of w: the
// linear layer x·wᵀ. Each block's values are decoded once and reused for
// all n rows, so it suits a few rows; for many, Dequantize the matrix and
// use the dense products.
func MicroLinearInto(y []float32, ldy int, x []float32, ldx int, n int, w *Microscaled) {
	if n <= 0 {
		return
	}
	if len(x) < (n-1)*ldx+w.cols || len(y) < (n-1)*ldy+w.rows {
		panic(fmt.Sprintf("ops: MicroLinearInto of %d rows over [%d, %d] has %d inputs and %d outputs", n, w.rows, w.cols, len(x), len(y)))
	}
	tasks := (w.rows + microRowsPerTask - 1) / microRowsPerTask
	workers := min(runtime.GOMAXPROCS(0), max(1, n*w.rows*w.cols/macsPerWorker))
	forEach(tasks, workers, func(t int) {
		var buf [32]float32
		var acc [16]float32
		r1 := min(w.rows, (t+1)*microRowsPerTask)
		for r := t * microRowsPerTask; r < r1; r++ {
			switch {
			case n == 1 && w.format == NVFP4:
				y[r] = w.rowDotNVFP4(x, r)
			case n == 1:
				y[r] = w.rowDotMXFP8(x, r)
			default:
				w.rowProducts(y[r:], ldy, x, ldx, n, r, &buf, &acc)
			}
		}
	})
}

// rowProducts sets y[i·ldy] to row i of x times row r of w, for i < n.
func (w *Microscaled) rowProducts(y []float32, ldy int, x []float32, ldx, n, r int, buf *[32]float32, acc *[16]float32) {
	bs := w.format.BlockSize()
	blocks := w.cols / bs
	for i0 := 0; i0 < n; i0 += len(acc) {
		i1 := min(n, i0+len(acc))
		clear(acc[:])
		for b := range blocks {
			vals := buf[:bs]
			// The block's values without its scale: partial sums times the
			// scale round differently from the dequantized weights, by
			// about one float32 ulp of each block's sum.
			if w.format == NVFP4 {
				for k, c := range w.values[(r*w.cols+b*bs)/2:][:bs/2] {
					vals[2*k], vals[2*k+1] = e2m1[c&0xF], e2m1[c>>4]
				}
			} else {
				for k, v := range w.values[r*w.cols+b*bs:][:bs] {
					vals[k] = e4m3[v]
				}
			}
			s := w.blockScale(r*blocks + b)
			for i := i0; i < i1; i++ {
				acc[i-i0] += dot16s(x[i*ldx+b*bs:], vals) * s
			}
		}
		for i := i0; i < i1; i++ {
			y[i*ldy] = acc[i-i0] * w.factor
		}
	}
}

// e2m1Pairs holds the values of both E2M1 codes of each byte, the low
// nibble first.
var e2m1Pairs = func() (t [256][2]float32) {
	for b := range 256 {
		t[b] = [2]float32{e2m1[b&0xF], e2m1[b>>4]}
	}
	return t
}()

// rowDotNVFP4 returns x[:Cols()] times row r of an NVFP4 matrix, as
// rowProducts computes it for one row of x but with four partial sums per
// block.
func (w *Microscaled) rowDotNVFP4(x []float32, r int) float32 {
	codes := w.values[r*w.cols/2 : (r+1)*w.cols/2]
	scales := w.scales[r*w.cols/16 : (r+1)*w.cols/16]
	x = x[:w.cols]
	var acc float32
	for b, s := range scales {
		c := (*[8]byte)(codes[b*8:])
		xs := (*[16]float32)(x[b*16:])
		p0, p1, p2, p3 := &e2m1Pairs[c[0]], &e2m1Pairs[c[1]], &e2m1Pairs[c[2]], &e2m1Pairs[c[3]]
		p4, p5, p6, p7 := &e2m1Pairs[c[4]], &e2m1Pairs[c[5]], &e2m1Pairs[c[6]], &e2m1Pairs[c[7]]
		s0 := xs[0]*p0[0] + xs[1]*p0[1] + xs[8]*p4[0] + xs[9]*p4[1]
		s1 := xs[2]*p1[0] + xs[3]*p1[1] + xs[10]*p5[0] + xs[11]*p5[1]
		s2 := xs[4]*p2[0] + xs[5]*p2[1] + xs[12]*p6[0] + xs[13]*p6[1]
		s3 := xs[6]*p3[0] + xs[7]*p3[1] + xs[14]*p7[0] + xs[15]*p7[1]
		acc += ((s0 + s1) + (s2 + s3)) * e4m3[s]
	}
	return acc * w.factor
}

// rowDotMXFP8 is rowDotNVFP4 for an MXFP8 matrix.
func (w *Microscaled) rowDotMXFP8(x []float32, r int) float32 {
	values := w.values[r*w.cols : (r+1)*w.cols]
	scales := w.scales[r*w.cols/32 : (r+1)*w.cols/32]
	x = x[:w.cols]
	var acc float32
	for b, s := range scales {
		v := (*[32]byte)(values[b*32:])
		xs := (*[32]float32)(x[b*32:])
		var s0, s1, s2, s3 float32
		for k := 0; k < 32; k += 4 {
			s0 += xs[k] * e4m3[v[k]]
			s1 += xs[k+1] * e4m3[v[k+1]]
			s2 += xs[k+2] * e4m3[v[k+2]]
			s3 += xs[k+3] * e4m3[v[k+3]]
		}
		acc += ((s0 + s1) + (s2 + s3)) * e8m0(s)
	}
	return acc
}

// dot16s returns x[:len(v)]·v for v of 16 or 32 values, in four partial
// sums.
func dot16s(x, v []float32) float32 {
	var s0, s1, s2, s3 float32
	for k := 0; k+16 <= len(v); k += 16 {
		xs, vs := (*[16]float32)(x[k:]), (*[16]float32)(v[k:])
		s0 += xs[0]*vs[0] + xs[4]*vs[4] + xs[8]*vs[8] + xs[12]*vs[12]
		s1 += xs[1]*vs[1] + xs[5]*vs[5] + xs[9]*vs[9] + xs[13]*vs[13]
		s2 += xs[2]*vs[2] + xs[6]*vs[6] + xs[10]*vs[10] + xs[14]*vs[14]
		s3 += xs[3]*vs[3] + xs[7]*vs[7] + xs[11]*vs[11] + xs[15]*vs[15]
	}
	return (s0 + s1) + (s2 + s3)
}
