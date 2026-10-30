package ops

import (
	"fmt"
	"math"
)

// RoPE is the rotary position embedding of Llama-family models, in Hugging
// Face's rotate-half convention: within each head, dimension j rotates with
// dimension j+headDim/2 by the angle position·invFreq[j].
type RoPE struct {
	headDim int
	invFreq []float32
}

// NewRoPE returns the default (unscaled) rotary embedding for heads of
// headDim with base theta. The inverse frequencies 1/theta^(2j/headDim) are
// computed in the float32 steps transformers takes; torch's float32 pow can
// differ from Go's correctly rounded one by an ulp (one of MiniCPM5's 64
// frequencies does).
func NewRoPE(headDim int, theta float64) *RoPE {
	if headDim <= 0 || headDim%2 != 0 {
		panic(fmt.Sprintf("ops.NewRoPE: head dimension %d is not even", headDim))
	}
	r := &RoPE{headDim: headDim, invFreq: make([]float32, headDim/2)}
	for j := range r.invFreq {
		e := float32(2*j) / float32(headDim)
		r.invFreq[j] = 1 / float32(math.Pow(theta, float64(e)))
	}
	return r
}

// InvFreq returns the inverse frequencies, one per pair of dimensions.
func (r *RoPE) InvFreq() []float32 { return r.invFreq }

// Rotation holds the cosines and sines for a run of positions, computed once
// per forward pass and applied to the queries and keys of every layer.
type Rotation struct {
	half, seq int
	cos, sin  []float32 // [seq, half]
}

// Rotation returns the rotation for seq positions starting at pos0. Angles
// are float32(position)·invFreq in float32, as torch computes them.
func (r *RoPE) Rotation(pos0, seq int) *Rotation {
	half := r.headDim / 2
	t := &Rotation{half: half, seq: seq, cos: make([]float32, seq*half), sin: make([]float32, seq*half)}
	for i := range seq {
		pos := float32(pos0 + i)
		for j, f := range r.invFreq {
			angle := float64(pos * f)
			t.cos[i*half+j], t.sin[i*half+j] = float32(math.Cos(angle)), float32(math.Sin(angle))
		}
	}
	return t
}

// Tables returns the cosines and sines, each [seq, headDim/2] row-major, for
// a GPU kernel to rotate with exactly the values Apply uses.
func (t *Rotation) Tables() (cos, sin []float32) { return t.cos, t.sin }

// Apply rotates the rotation's seq rows of x in place: row i at x[i*ld],
// heads heads of 2·half values each. Each value is computed as torch's
// x·cos + rotate_half(x)·sin: two rounded products, then a rounded sum.
func (t *Rotation) Apply(x []float32, ld, heads int) {
	half, hd := t.half, 2*t.half
	if t.seq > 0 {
		_ = x[(t.seq-1)*ld+heads*hd-1]
	}
	for i := range t.seq {
		cos, sin := t.cos[i*half:(i+1)*half], t.sin[i*half:(i+1)*half]
		row := x[i*ld : i*ld+heads*hd]
		for h := range heads {
			lo, hi := row[h*hd:h*hd+half], row[h*hd+half:(h+1)*hd]
			for j, c := range cos {
				x1, x2, s := lo[j], hi[j], sin[j]
				// The conversions keep each product rounded on its own, as
				// torch's separate kernels round it, instead of fused.
				lo[j] = float32(x1*c) + float32(-x2*s)
				hi[j] = float32(x2*c) + float32(x1*s)
			}
		}
	}
}
