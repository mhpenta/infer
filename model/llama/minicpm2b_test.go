package llama

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
)

// TestMiniCPM5For2BOnMetal runs MiniCPM5-2B (5 GB, not in CI:
// scripts/fetch_models.sh minicpm5-2b) on the GPU and on the CPU, which was
// checked against PyTorch when the model was added, and requires the same
// greedy tokens with every step's logits within relTol of the CPU's. The
// prompts are MiniCPM5-1B's golden ones, which share its tokenizer.
func TestMiniCPM5For2BOnMetal(t *testing.T) {
	if len(testDevices(t)) < 2 {
		t.Skip("no Metal device")
	}
	if testing.Short() {
		t.Skip("5 GB on each device, skipped with -short")
	}
	dir := golden.ModelDir(t, "INFER_MINICPM5_2B_DIR", "minicpm5-2b")
	var gm struct {
		Cases []modelCase `json:"cases"`
	}
	golden.JSON(t, "minicpm5_1b_model.json", &gm)
	const steps = 24

	type step struct {
		token  int
		logits []float32
	}
	run := func(device string) [][]step {
		m, err := LoadWith(dir, Options{Device: device})
		if err != nil {
			t.Fatal(err)
		}
		defer func() { m.Close(); runtime.GC() }()
		var all [][]step
		for i, c := range gm.Cases {
			if i == longCase {
				continue // a 2,461-token prompt takes the CPU a minute here
			}
			var got []step
			if err := m.GenerateFunc(c.InputIDs, steps, func(token int, logits []float32) bool {
				got = append(got, step{token, slices.Clone(logits)})
				return true
			}); err != nil {
				t.Fatal(err)
			}
			all = append(all, got)
		}
		return all
	}
	cpu, gpu := run(accel.CPU), run(accel.Metal)
	for i := range cpu {
		t.Run(fmt.Sprintf("case%d", i), func(t *testing.T) {
			worst := 0.0
			for s := range cpu[i] {
				if s >= len(gpu[i]) || gpu[i][s].token != cpu[i][s].token {
					t.Fatalf("step %d: Metal generated a different token", s)
				}
				var peak, d float64
				for j, v := range cpu[i][s].logits {
					peak = max(peak, math.Abs(float64(v)))
					d = max(d, math.Abs(float64(gpu[i][s].logits[j])-float64(v)))
				}
				if d/peak > relTol {
					t.Fatalf("step %d: logits differ from the CPU's by %.2g of the row's range", s, d/peak)
				}
				worst = max(worst, d/peak)
			}
			t.Logf("%d tokens, all the CPU's; largest logit difference %.2g of a row's range", len(cpu[i]), worst)
		})
	}
}
