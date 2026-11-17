package ops

import "fmt"

// DeltaRuleArgs describes gated delta-rule linear attention (Gated
// DeltaNet), Qwen3.5's linear-attention layers: KeyHeads heads of KeyDim for
// the queries and keys, ValueHeads heads of ValueDim for the values, each
// group of ValueHeads/KeyHeads value heads reading one key head.
type DeltaRuleArgs struct {
	KeyHeads, ValueHeads int
	KeyDim, ValueDim     int
}

// StateSize returns the length of a recurrent state: one KeyDim×ValueDim
// matrix per value head.
func (a DeltaRuleArgs) StateSize() int { return a.ValueHeads * a.KeyDim * a.ValueDim }

// GatedDeltaRule runs seq positions of gated delta-rule linear attention,
// updating state (StateSize values, zero for an empty prefix) in place. Row
// i of q, k and v is at q[i*ldq], k[i*ldk] and v[i*ldv]: KeyHeads heads of
// KeyDim for q and k, already L2-normalized (q also scaled), and ValueHeads
// heads of ValueDim for v. decay[i*ValueHeads+h] is position i's decay
// exp(g) for value head h, and beta[i*ValueHeads+h] its write strength. Row
// i of the output, ValueHeads heads of ValueDim, goes to out[i*ldOut].
//
// For each position and value head, with S the head's state (key by value):
//
//	S ← decay·S
//	S ← S + k ⊗ (beta·(v − Sᵀk))
//	o ← Sᵀq
//
// the recurrence transformers' torch_recurrent_gated_delta_rule runs, one
// position at a time. Each head's positions run in order and heads run in
// parallel, so a position's output does not depend on how many positions
// run with it.
func GatedDeltaRule(out []float32, ldOut int, q []float32, ldq int, k []float32, ldk int, v []float32, ldv int, decay, beta, state []float32, seq int, args DeltaRuleArgs) {
	nk, nv, dk, dv := args.KeyHeads, args.ValueHeads, args.KeyDim, args.ValueDim
	if nk <= 0 || nv <= 0 || nv%nk != 0 || dk <= 0 || dv <= 0 {
		panic(fmt.Sprintf("ops.GatedDeltaRule: %d value heads do not share %d key heads (dims %d, %d)", nv, nk, dk, dv))
	}
	if len(state) != args.StateSize() || len(decay) < seq*nv || len(beta) < seq*nv {
		panic(fmt.Sprintf("ops.GatedDeltaRule: state has %d values, want %d; decay %d and beta %d for %d positions of %d heads",
			len(state), args.StateSize(), len(decay), len(beta), seq, nv))
	}
	if seq == 0 {
		return
	}
	_ = q[(seq-1)*ldq+nk*dk-1]
	_ = k[(seq-1)*ldk+nk*dk-1]
	_ = v[(seq-1)*ldv+nv*dv-1]
	_ = out[(seq-1)*ldOut+nv*dv-1]
	group := nv / nk
	forEach(nv, workersFor(3*seq*nv*dk*dv), func(h int) {
		s := state[h*dk*dv : (h+1)*dk*dv]
		kh := h / group
		mem := make([]float32, dv)
		for i := range seq {
			qi := q[i*ldq+kh*dk : i*ldq+(kh+1)*dk]
			ki := k[i*ldk+kh*dk : i*ldk+(kh+1)*dk]
			vi := v[i*ldv+h*dv : i*ldv+(h+1)*dv]
			oi := out[i*ldOut+h*dv : i*ldOut+(h+1)*dv]
			d, b := decay[i*nv+h], beta[i*nv+h]

			// Decay the state and read what it already holds for k.
			clear(mem)
			for r, kr := range ki {
				row := s[r*dv : (r+1)*dv]
				for c := range row {
					x := row[c] * d
					row[c] = x
					mem[c] += x * kr
				}
			}
			// Write the correction, then read the output for q.
			for c := range mem {
				mem[c] = (vi[c] - mem[c]) * b
			}
			clear(oi)
			for r, kr := range ki {
				row, qr := s[r*dv:(r+1)*dv], qi[r]
				for c, delta := range mem {
					x := row[c] + kr*delta
					row[c] = x
					oi[c] += x * qr
				}
			}
		}
	})
}
