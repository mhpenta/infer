package llama

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/ops"
)

// BenchmarkDecode times greedy decoding steps (one token through the model,
// the output layer and argmax) after each model's 65- or 66-token summary
// prompt, as generation runs them, on each device. Prefill is not timed.
func BenchmarkDecode(b *testing.B) {
	for _, g := range goldenModels {
		for _, device := range testDevices(b) {
			g := g.on(device)
			b.Run(g.key(), func(b *testing.B) {
				m, cases, _ := g.load(b)
				defer g.release()
				benchDecode(b, m, cases[2].InputIDs)
			})
		}
	}
}

// benchDecode times greedy decoding steps after prompt.
func benchDecode(b *testing.B, m *Model, prompt []int) {
	const steps = 32
	var elapsed time.Duration
	for range b.N {
		st := m.newState(len(prompt) + steps)
		logits, err := st.next(prompt)
		start := time.Now()
		for range steps {
			if err != nil {
				b.Fatal(err)
			}
			logits, err = st.next([]int{ops.Argmax(logits)})
		}
		elapsed += time.Since(start)
		st.release()
	}
	perToken := elapsed / time.Duration(b.N*steps)
	b.ReportMetric(float64(perToken.Microseconds())/1e3, "ms/token")
	b.ReportMetric(1/perToken.Seconds(), "tokens/s")
}

// BenchmarkDecodeWeights times decoding MiniCPM5-1B and MiniCPM5-2B (when
// fetched) with bf16 and int8 weights, on each device, after MiniCPM5-1B's
// 65-token summary prompt (the models share a tokenizer).
func BenchmarkDecodeWeights(b *testing.B) {
	_, cases, _ := miniCPM.load(b)
	prompt := cases[2].InputIDs
	miniCPM.release()
	for _, model := range []struct{ env, dir string }{{"INFER_MINICPM5_DIR", "minicpm5-1b"}, {"INFER_MINICPM5_2B_DIR", "minicpm5-2b"}} {
		for _, device := range testDevices(b) {
			for _, int8 := range []bool{false, true} {
				name := fmt.Sprintf("%s/%s/%s", model.dir, device, map[bool]string{false: "bf16", true: "int8"}[int8])
				b.Run(name, func(b *testing.B) {
					m, err := LoadWith(golden.ModelDir(b, model.env, model.dir), Options{Device: device, Int8: int8})
					if err != nil {
						b.Fatal(err)
					}
					defer func() { m.Close(); runtime.GC() }()
					benchDecode(b, m, prompt)
				})
			}
		}
	}
}

// BenchmarkPrefill times running each model's prompts through an empty
// cache to the first token's logits: the 65- or 66-token summary prompt and
// the long one (2,461 tokens for MiniCPM5-1B), on each device.
func BenchmarkPrefill(b *testing.B) {
	for _, g := range goldenModels {
		for _, device := range testDevices(b) {
			g := g.on(device)
			for _, c := range []int{2, longCase} {
				b.Run(g.key()+"/"+map[int]string{2: "summary", longCase: "long"}[c], func(b *testing.B) {
					m, cases, _ := g.load(b)
					prompt := cases[c].InputIDs
					for b.Loop() {
						st := m.newState(len(prompt))
						if _, err := st.next(prompt); err != nil {
							b.Fatal(err)
						}
						st.release()
					}
					per := b.Elapsed() / time.Duration(b.N)
					b.ReportMetric(float64(per.Milliseconds()), "ms")
					b.ReportMetric(float64(len(prompt))/per.Seconds(), "tokens/s")
					b.ReportMetric(0, "ns/op")
				})
			}
			g.release()
		}
	}
}
