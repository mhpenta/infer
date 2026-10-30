package llama

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/tensor"
)

// int8Model loads MiniCPM5-1B with int8 weights on device.
func int8Model(t *testing.T, device string) *Model {
	t.Helper()
	m, err := LoadWith(golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b"), Options{Int8: true, Device: device})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(m.Device(), device) {
		t.Fatalf("int8 weights loaded on %s, want %s", m.Device(), device)
	}
	quantized := m.lmHead != nil && m.lmHead.Int8() && m.layers[0].gateUp.Int8() && m.embed.bf16 != nil
	if g := m.gpu; g != nil {
		quantized = g.lmHead.Type == metal.Int8 && g.layers[0].gateUp.Type == metal.Int8 && g.embedType != metal.Int8
	}
	if !quantized {
		t.Fatal("int8 model does not hold int8 linear layers and bf16 embeddings")
	}
	return m
}

// agreement is how closely one model's next-token distributions follow a
// reference model's over a sequence.
type agreement struct {
	positions int
	sameTop   int     // positions where both rank the same token first
	kl        float64 // summed KL(reference || model), nats
	nll, ref  float64 // summed negative log-likelihood of the actual next tokens
}

func (a agreement) String() string {
	n := float64(a.positions)
	return fmt.Sprintf("%d positions: top-1 agrees %.2f%%, mean KL %.2g nats, perplexity %.4g vs %.4g (%+.2f%%)",
		a.positions, 100*float64(a.sameTop)/n, a.kl/n, math.Exp(a.nll/n), math.Exp(a.ref/n), 100*(math.Exp((a.nll-a.ref)/n)-1))
}

// compareModels runs both models teacher-forced over seq and compares their
// distributions at every position; next[p] is the token that follows
// position p.
func compareModels(model, ref *Model, seq, next []int) agreement {
	y, yr := forwardAll(model, seq), forwardAll(ref, seq)
	const chunk = 64
	vocab := model.Config.VocabSize
	var a agreement
	for p0 := 0; p0 < len(seq); p0 += chunk {
		rows := min(chunk, len(seq)-p0)
		lg, lr := tensor.New(rows, vocab), tensor.New(rows, vocab)
		model.LogitsInto(lg, tensor.FromData(y.Data[p0*y.Cols():(p0+rows)*y.Cols()], rows, y.Cols()))
		ref.LogitsInto(lr, tensor.FromData(yr.Data[p0*yr.Cols():(p0+rows)*yr.Cols()], rows, yr.Cols()))
		for r := range rows {
			g, w := logSoftmax(lg.Row(r)), logSoftmax(lr.Row(r))
			if argmax64(g) == argmax64(w) {
				a.sameTop++
			}
			for i, lw := range w {
				a.kl += math.Exp(lw) * (lw - g[i])
			}
			a.nll -= g[next[p0+r]]
			a.ref -= w[next[p0+r]]
			a.positions++
		}
	}
	return a
}

func logSoftmax(x []float32) []float64 {
	peak := float64(slices.Max(x))
	sum := 0.0
	for _, v := range x {
		sum += math.Exp(float64(v) - peak)
	}
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v) - peak - math.Log(sum)
	}
	return out
}

func argmax64(x []float64) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// TestInt8MatchesBF16 checks MiniCPM5-1B with int8 weights against the same
// model in bf16 (which matches PyTorch) on every golden sequence, on each
// device: the top token must agree at most positions, the distributions
// must stay close (on MiniCPM5-1B: 94-100% agreement and a mean KL
// divergence of 0.0003-0.007 nats, with no measurable change from keeping
// the output layer in bf16), and greedy generation must give PyTorch's
// tokens.
func TestInt8MatchesBF16(t *testing.T) {
	for _, device := range testDevices(t) {
		t.Run(device, func(t *testing.T) { checkInt8(t, device) })
	}
}

func checkInt8(t *testing.T, device string) {
	g := miniCPM.on(device)
	m, cases, _ := g.load(t)
	defer g.release()
	q := int8Model(t, device)
	defer func() { q.Close(); runtime.GC() }()
	for i, c := range cases {
		t.Run(fmt.Sprintf("case%d_%d_in", i, len(c.InputIDs)), func(t *testing.T) {
			if i == longCase && testing.Short() {
				t.Skip("long prompt skipped with -short")
			}
			seq := c.sequence()
			next := append(slices.Clone(seq[1:]), c.OutputIDs[len(c.OutputIDs)-1])
			a := compareModels(q, m, seq, next)
			t.Log(a)
			if same := float64(a.sameTop) / float64(a.positions); same < 0.9 {
				t.Errorf("top-1 agrees at %.1f%% of positions, want >= 90%%", 100*same)
			}
			if kl := a.kl / float64(a.positions); kl > 0.02 {
				t.Errorf("mean KL divergence %.3g nats, want <= 0.02", kl)
			}
			got, err := q.Generate(c.InputIDs, len(c.OutputIDs))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, c.OutputIDs) {
				t.Errorf("greedy int8 generated %v\nwant %v", got, c.OutputIDs)
			}
		})
	}
}
