package ops

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// deltaRuleReference is GatedDeltaRule in float64, written as the formulas.
func deltaRuleReference(q, k, v, decay, beta []float32, state []float64, seq int, a DeltaRuleArgs) []float64 {
	nk, nv, dk, dv := a.KeyHeads, a.ValueHeads, a.KeyDim, a.ValueDim
	out := make([]float64, seq*nv*dv)
	for i := range seq {
		for h := range nv {
			kh := h / (nv / nk)
			s := state[h*dk*dv : (h+1)*dk*dv]
			qi, ki := q[i*nk*dk+kh*dk:], k[i*nk*dk+kh*dk:]
			vi := v[i*nv*dv+h*dv:]
			for j := range s {
				s[j] *= float64(decay[i*nv+h])
			}
			for c := range dv {
				var mem float64
				for r := range dk {
					mem += s[r*dv+c] * float64(ki[r])
				}
				delta := (float64(vi[c]) - mem) * float64(beta[i*nv+h])
				for r := range dk {
					s[r*dv+c] += float64(ki[r]) * delta
				}
			}
			for c := range dv {
				var o float64
				for r := range dk {
					o += s[r*dv+c] * float64(qi[r])
				}
				out[i*nv*dv+h*dv+c] = o
			}
		}
	}
	return out
}

// randomDeltaInputs returns inputs shaped like Qwen3.5's: unit-length keys,
// queries of length 1/sqrt(KeyDim), decays in (0, 1] and betas in (0, 1).
func randomDeltaInputs(r *rand.Rand, seq int, a DeltaRuleArgs) (q, k, v, decay, beta []float32) {
	unit := func(n, rows int, scale float64) []float32 {
		x := make([]float32, rows*n)
		for i := 0; i < len(x); i += n {
			var sum float64
			for j := range n {
				x[i+j] = float32(r.NormFloat64())
				sum += float64(x[i+j]) * float64(x[i+j])
			}
			for j := range n {
				x[i+j] = float32(float64(x[i+j]) / math.Sqrt(sum) * scale)
			}
		}
		return x
	}
	q = unit(a.KeyDim, seq*a.KeyHeads, 1/math.Sqrt(float64(a.KeyDim)))
	k = unit(a.KeyDim, seq*a.KeyHeads, 1)
	v = make([]float32, seq*a.ValueHeads*a.ValueDim)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	decay, beta = make([]float32, seq*a.ValueHeads), make([]float32, seq*a.ValueHeads)
	for i := range decay {
		decay[i] = float32(math.Exp(-r.Float64() * 2))
		beta[i] = float32(r.Float64())
	}
	return q, k, v, decay, beta
}

func TestGatedDeltaRuleMatchesReference(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	for _, a := range []DeltaRuleArgs{
		{KeyHeads: 2, ValueHeads: 2, KeyDim: 16, ValueDim: 8},
		{KeyHeads: 2, ValueHeads: 6, KeyDim: 8, ValueDim: 16}, // three value heads per key head
		{KeyHeads: 4, ValueHeads: 8, KeyDim: 128, ValueDim: 128},
	} {
		const seq = 37
		q, k, v, decay, beta := randomDeltaInputs(r, seq, a)
		state := make([]float32, a.StateSize())
		out := make([]float32, seq*a.ValueHeads*a.ValueDim)
		GatedDeltaRule(out, a.ValueHeads*a.ValueDim, q, a.KeyHeads*a.KeyDim, k, a.KeyHeads*a.KeyDim, v, a.ValueHeads*a.ValueDim, decay, beta, state, seq, a)

		refState := make([]float64, a.StateSize())
		want := deltaRuleReference(q, k, v, decay, beta, refState, seq, a)
		for i, w := range want {
			if d := math.Abs(float64(out[i]) - w); d > 1e-5*(1+math.Abs(w)) {
				t.Fatalf("%+v: out[%d] = %g, want %g", a, i, out[i], w)
			}
		}
		for i, w := range refState {
			if d := math.Abs(float64(state[i]) - w); d > 1e-5*(1+math.Abs(w)) {
				t.Fatalf("%+v: state[%d] = %g, want %g", a, i, state[i], w)
			}
		}
	}
}

// TestGatedDeltaRuleSplitMatchesWhole runs a sequence in one call and in
// pieces carrying the state, as a cached prefix does: every bit must match.
func TestGatedDeltaRuleSplitMatchesWhole(t *testing.T) {
	r := rand.New(rand.NewPCG(43, 44))
	a := DeltaRuleArgs{KeyHeads: 2, ValueHeads: 4, KeyDim: 32, ValueDim: 16}
	const seq = 20
	ld := a.ValueHeads * a.ValueDim
	q, k, v, decay, beta := randomDeltaInputs(r, seq, a)

	whole, wholeState := make([]float32, seq*ld), make([]float32, a.StateSize())
	GatedDeltaRule(whole, ld, q, a.KeyHeads*a.KeyDim, k, a.KeyHeads*a.KeyDim, v, ld, decay, beta, wholeState, seq, a)

	parts, partState := make([]float32, seq*ld), make([]float32, a.StateSize())
	ldk := a.KeyHeads * a.KeyDim
	for _, cut := range [][2]int{{0, 7}, {7, 8}, {8, 20}} {
		i, j := cut[0], cut[1]
		GatedDeltaRule(parts[i*ld:], ld, q[i*ldk:], ldk, k[i*ldk:], ldk, v[i*ld:], ld,
			decay[i*a.ValueHeads:], beta[i*a.ValueHeads:], partState, j-i, a)
	}
	if !slices.Equal(parts, whole) || !slices.Equal(partState, wholeState) {
		t.Fatal("running in pieces changed the output or the state")
	}
}
