package ops

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

// refAttentionArgs is attention with bias and a causal mask, from the
// definition, in float64, on dense [seq, hidden] and [kvSeq, hidden] inputs.
func refAttentionArgs(q, k, v []float32, seq, kvSeq, hidden int, args AttentionArgs) []float64 {
	dh := hidden / args.Heads
	out := make([]float64, seq*hidden)
	for h := range args.Heads {
		off := h * dh
		for i := range seq {
			scores := make([]float64, kvSeq)
			peak := math.Inf(-1)
			for j := range kvSeq {
				if args.Causal && j > args.Offset+i {
					scores[j] = math.Inf(-1)
					continue
				}
				for d := range dh {
					scores[j] += float64(q[i*hidden+off+d]) * float64(k[j*hidden+off+d])
				}
				scores[j] *= float64(args.Scale)
				if args.Bias != nil {
					scores[j] += float64(args.Bias[(h*seq+i)*kvSeq+j])
				}
				peak = max(peak, scores[j])
			}
			var sum float64
			for j := range scores {
				scores[j] = math.Exp(scores[j] - peak)
				sum += scores[j]
			}
			for j := range kvSeq {
				for d := range dh {
					out[i*hidden+off+d] += scores[j] / sum * float64(v[j*hidden+off+d])
				}
			}
		}
	}
	return out
}

func TestAttentionStridedBiasAndCausal(t *testing.T) {
	r := rand.New(rand.NewPCG(51, 52))
	tests := []struct {
		seq, kvSeq, heads, headDim, offset int
		causal, bias                       bool
	}{
		{5, 5, 2, 8, 0, false, true},  // encoder: bias, no mask
		{7, 7, 4, 16, 0, true, true},  // decoder, whole sequence
		{1, 9, 4, 16, 8, true, true},  // decoder step with 8 cached positions
		{3, 11, 2, 8, 8, true, false}, // several new positions after a cache
		{6, 13, 2, 8, 0, false, false},
	}
	for _, tt := range tests {
		name := fmt.Sprintf("seq%d_kv%d_off%d_causal%v_bias%v", tt.seq, tt.kvSeq, tt.offset, tt.causal, tt.bias)
		t.Run(name, func(t *testing.T) {
			hidden := tt.heads * tt.headDim
			q := randomSlice(r, tt.seq*hidden)
			k := randomSlice(r, tt.kvSeq*hidden)
			v := randomSlice(r, tt.kvSeq*hidden)
			args := AttentionArgs{Heads: tt.heads, Scale: 0.5, Causal: tt.causal, Offset: tt.offset}
			if tt.bias {
				args.Bias = randomSlice(r, tt.heads*tt.seq*tt.kvSeq)
			}
			want := refAttentionArgs(q, k, v, tt.seq, tt.kvSeq, hidden, args)

			// k and v in separate strided buffers (as a KV cache holds
			// them), and out wider than hidden.
			kv := make([]float32, tt.kvSeq*2*hidden)
			for j := range tt.kvSeq {
				copy(kv[j*2*hidden:], k[j*hidden:(j+1)*hidden])
				copy(kv[j*2*hidden+hidden:], v[j*hidden:(j+1)*hidden])
			}
			ldOut := hidden + 3
			out := make([]float32, tt.seq*ldOut)
			AttentionStrided(out, ldOut, q, hidden, kv, 2*hidden, kv[hidden:], 2*hidden, tt.seq, tt.kvSeq, hidden, args)
			for i := range tt.seq {
				for c := range hidden {
					g, w := float64(out[i*ldOut+c]), want[i*hidden+c]
					if math.Abs(g-w) > 1e-5*(1+math.Abs(w)) {
						t.Fatalf("out[%d][%d] = %g, want %g", i, c, g, w)
					}
				}
			}
		})
	}
}

func TestRMSNorm(t *testing.T) {
	x := tensor.FromData([]float32{3, 4, 0, -4, 3, 0}, 2, 3)
	w := tensor.FromData([]float32{1, 2, 3}, 3)
	RMSNorm(x, w, 0)
	// Both rows have root mean square √(25/3).
	s := float32(math.Sqrt(3.0 / 25))
	nearAll(t, x.Data, []float32{3 * s, 8 * s, 0, -4 * s, 6 * s, 0})
}

func TestReLUAndArgmax(t *testing.T) {
	x := []float32{-1, 2, -0.5, 7, 7, 0}
	ReLU(x)
	nearAll(t, x, []float32{0, 2, 0, 7, 7, 0})
	if got := Argmax(x); got != 3 {
		t.Errorf("Argmax = %d, want 3 (the first of the tied maxima)", got)
	}
}

// TestKVCacheMatchesAttentionStrided checks that attention over a KVCache
// is bit-identical to AttentionStrided on the same keys and values, however
// the cache was filled (one position at a time from capacity 1, so it grows
// repeatedly, or all at once).
func TestKVCacheMatchesAttentionStrided(t *testing.T) {
	r := rand.New(rand.NewPCG(53, 54))
	tests := []struct {
		heads, headDim, kvSeq, seq, offset int
		causal, bias                       bool
	}{
		{8, 64, 1, 1, 0, true, true},
		{8, 64, 37, 1, 36, true, true}, // a decode step
		{4, 24, 50, 3, 47, true, false},
		{2, 8, 70, 5, 0, false, true}, // cross-attention style
		{3, 16, 17, 17, 0, true, true},
	}
	for _, tt := range tests {
		name := fmt.Sprintf("h%d_d%d_kv%d_seq%d_off%d", tt.heads, tt.headDim, tt.kvSeq, tt.seq, tt.offset)
		t.Run(name, func(t *testing.T) {
			hidden := tt.heads * tt.headDim
			q := randomSlice(r, tt.seq*hidden)
			kv := randomSlice(r, tt.kvSeq*2*hidden) // keys then values per row
			args := AttentionArgs{Heads: tt.heads, Scale: 1, Causal: tt.causal, Offset: tt.offset}
			if tt.bias {
				args.Bias = randomSlice(r, tt.heads*tt.seq*tt.kvSeq)
			}
			want := make([]float32, tt.seq*hidden)
			AttentionStrided(want, hidden, q, hidden, kv, 2*hidden, kv[hidden:], 2*hidden, tt.seq, tt.kvSeq, hidden, args)

			oneByOne := NewKVCache(tt.heads, tt.headDim, 1)
			for j := range tt.kvSeq {
				row := kv[j*2*hidden:]
				oneByOne.Append(row, 2*hidden, row[hidden:], 2*hidden, 1)
			}
			atOnce := NewKVCache(tt.heads, tt.headDim, tt.kvSeq)
			atOnce.Append(kv, 2*hidden, kv[hidden:], 2*hidden, tt.kvSeq)
			chunked := NewKVCache(tt.heads, tt.headDim, 2)
			for j := 0; j < tt.kvSeq; j += 7 {
				n := min(7, tt.kvSeq-j)
				chunked.Append(kv[j*2*hidden:], 2*hidden, kv[j*2*hidden+hidden:], 2*hidden, n)
			}
			for _, c := range []struct {
				name  string
				cache *KVCache
			}{{"one by one", oneByOne}, {"at once", atOnce}, {"chunked", chunked}} {
				if c.cache.Len() != tt.kvSeq {
					t.Fatalf("%s: Len %d, want %d", c.name, c.cache.Len(), tt.kvSeq)
				}
				got := make([]float32, tt.seq*hidden)
				c.cache.Attend(got, hidden, q, hidden, tt.seq, args)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("%s: out[%d] = %g, AttentionStrided gives %g", c.name, i, got[i], want[i])
					}
				}
			}
		})
	}
}
