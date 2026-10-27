package ops

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
)

// TestRoPEInvFreqMatchesTorch compares MiniCPM5's inverse frequencies
// (head dimension 128, theta 5e6) with the ones transformers computed: equal
// but for at most one that torch's float32 pow rounds the other way.
func TestRoPEInvFreqMatchesTorch(t *testing.T) {
	f, err := safetensors.Open(golden.Path("minicpm5_1b_model.safetensors"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	ref, err := f.Tensor("rope.inv_freq")
	if err != nil {
		t.Fatal(err)
	}
	got := NewRoPE(128, 5e6).InvFreq()
	if len(got) != ref.Size() {
		t.Fatalf("%d frequencies, want %d", len(got), ref.Size())
	}
	differ := 0
	for j, want := range ref.Data {
		if d := int64(math.Float32bits(got[j])) - int64(math.Float32bits(want)); d != 0 {
			differ++
			if d < -1 || d > 1 {
				t.Errorf("inv_freq[%d] = %g, torch %g: %d ulps apart", j, got[j], want, d)
			}
		}
	}
	if differ > 1 {
		t.Errorf("%d frequencies differ from torch's, want at most 1", differ)
	}
}

// TestRoPEMatchesReference rotates random heads and checks every value
// against the rotation computed in float64 from the same float32 angles.
func TestRoPEMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(71, 72))
	const headDim, heads, seq, pos0, ld = 16, 3, 5, 2000, 3*16 + 7
	rope := NewRoPE(headDim, 10000)
	x := randomSlice(r, seq*ld)
	orig := append([]float32(nil), x...)
	rope.Rotation(pos0, seq).Apply(x, ld, heads)
	half := headDim / 2
	for i := range seq {
		for h := range heads {
			base := i*ld + h*headDim
			for j := range half {
				angle := float64(float32(pos0+i) * rope.invFreq[j])
				c, s := math.Cos(angle), math.Sin(angle)
				x1, x2 := float64(orig[base+j]), float64(orig[base+j+half])
				for _, w := range []struct {
					got  float32
					want float64
				}{{x[base+j], x1*c - x2*s}, {x[base+j+half], x2*c + x1*s}} {
					if math.Abs(float64(w.got)-w.want) > 1e-6*(math.Abs(x1)+math.Abs(x2)) {
						t.Fatalf("row %d head %d dim %d: %g, want %g", i, h, j, w.got, w.want)
					}
				}
			}
		}
	}
	// Columns past the heads are untouched.
	for i := range seq {
		for c := heads * headDim; c < ld; c++ {
			if x[i*ld+c] != orig[i*ld+c] {
				t.Fatalf("row %d column %d changed", i, c)
			}
		}
	}
}

// TestRoPERotationOffset checks that rotating one position at a time, as
// decoding does, gives the same bits as rotating a whole sequence.
func TestRoPERotationOffset(t *testing.T) {
	r := rand.New(rand.NewPCG(73, 74))
	const headDim, heads, seq = 128, 2, 40
	rope := NewRoPE(headDim, 5e6)
	x := randomSlice(r, seq*heads*headDim)
	steps := append([]float32(nil), x...)
	rope.Rotation(0, seq).Apply(x, heads*headDim, heads)
	for i := range seq {
		rope.Rotation(i, 1).Apply(steps[i*heads*headDim:], heads*headDim, heads)
	}
	if !bitsEqual(x, steps) {
		t.Fatal("one position at a time differs from the whole sequence")
	}
}

// TestRoPERelativePosition checks RoPE's defining property: the dot product
// of a rotated query and key depends only on their distance.
func TestRoPERelativePosition(t *testing.T) {
	r := rand.New(rand.NewPCG(75, 76))
	const headDim = 64
	rope := NewRoPE(headDim, 10000)
	q, k := randomSlice(r, headDim), randomSlice(r, headDim)
	dot := func(m, n int) float64 {
		qm, kn := append([]float32(nil), q...), append([]float32(nil), k...)
		rope.Rotation(m, 1).Apply(qm, headDim, 1)
		rope.Rotation(n, 1).Apply(kn, headDim, 1)
		var s float64
		for i := range qm {
			s += float64(qm[i]) * float64(kn[i])
		}
		return s
	}
	near, far := dot(7, 3), dot(1007, 1003)
	if math.Abs(near-far) > 1e-3*(1+math.Abs(near)) {
		t.Errorf("q·k at distance 4: %g near the start, %g 1000 positions later", near, far)
	}
}
