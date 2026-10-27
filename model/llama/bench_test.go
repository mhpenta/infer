package llama

import (
	"testing"
	"time"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// BenchmarkDecode times greedy decoding steps (one token through the model,
// the output layer and argmax) after each model's 65- or 66-token summary
// prompt, as generation runs them. Prefill is not timed.
func BenchmarkDecode(b *testing.B) {
	const steps = 32
	for _, g := range goldenModels {
		b.Run(g.dir, func(b *testing.B) {
			m, cases, _ := g.load(b)
			defer g.release()
			prompt := cases[2].InputIDs
			logits := tensor.New(1, m.Config.VocabSize)
			var elapsed time.Duration
			for range b.N {
				st := m.newState(len(prompt) + steps)
				y := st.forward(prompt, nil)
				start := time.Now()
				for range steps {
					m.LogitsInto(logits, tensor.FromData(y.Row(y.Rows()-1), 1, y.Cols()))
					y = st.forward([]int{ops.Argmax(logits.Data)}, nil)
				}
				elapsed += time.Since(start)
			}
			perToken := elapsed / time.Duration(b.N*steps)
			b.ReportMetric(float64(perToken.Microseconds())/1e3, "ms/token")
			b.ReportMetric(1/perToken.Seconds(), "tokens/s")
		})
	}
}
