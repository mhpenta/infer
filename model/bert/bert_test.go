package bert

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/pool"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tokenizer/wordpiece"
)

// Tolerances for float32 differences against PyTorch. Each layer's output is
// compared, so a drift is pinned to the layer where it starts.
const (
	atol      = 1e-4
	rtol      = 1e-4
	minCosine = 0.9999
)

// testDevices returns the devices to test on: the CPU, and Metal when this
// machine has it (or must, with INFER_REQUIRE_METAL=1).
func testDevices(t testing.TB) []string {
	_, err := accel.OpenMetal()
	if errors.Is(err, metal.ErrUnavailable) && os.Getenv("INFER_REQUIRE_METAL") != "1" {
		return []string{accel.CPU}
	}
	if err != nil {
		t.Fatal(err)
	}
	return []string{accel.CPU, accel.Metal}
}

// loadOn loads the model in dir on device, closing it when the test ends.
func loadOn(t testing.TB, dir, device string) *Model {
	t.Helper()
	model, err := LoadWith(dir, Options{Device: device})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(model.Device(), device) {
		t.Fatalf("loaded on %s, want %s", model.Device(), device)
	}
	t.Cleanup(func() { model.Close() })
	return model
}

// TestMiniLMMatchesPython checks every layer's output and the pooled
// embedding against PyTorch, on each device.
func TestMiniLMMatchesPython(t *testing.T) {
	for _, device := range testDevices(t) {
		t.Run(device, func(t *testing.T) { checkMiniLM(t, device) })
	}
}

func checkMiniLM(t *testing.T, device string) {
	dir := golden.ModelDir(t, "INFER_MINILM_DIR", "all-MiniLM-L6-v2")
	model := loadOn(t, dir, device)
	tok, err := wordpiece.LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	want, err := safetensors.Open(golden.Path("minilm_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()

	for i, c := range golden.Cases(t, "minilm_model.json") {
		t.Run(fmt.Sprintf("case%d_%d_tokens", i, len(c.IDs)), func(t *testing.T) {
			if ids := tok.Encode(c.Text); !slices.Equal(ids, c.IDs) {
				t.Fatalf("tokenizer mismatch for %q", c.Text)
			}
			states, err := model.HiddenStates(c.IDs)
			if err != nil {
				t.Fatal(err)
			}
			if len(states) != model.Config.NumHiddenLayers+1 {
				t.Fatalf("got %d hidden states, want %d", len(states), model.Config.NumHiddenLayers+1)
			}
			var lastDiff float64
			for j, got := range states {
				ref, err := want.Tensor(fmt.Sprintf("case%d.hidden.%d", i, j))
				if err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(got.Shape, ref.Shape) {
					t.Fatalf("hidden state %d: shape %v, want %v", j, got.Shape, ref.Shape)
				}
				if k, diff := firstMismatch(got.Data, ref.Data); k >= 0 {
					t.Fatalf("hidden state %d (0 = embeddings): element %d (token %d, dim %d) differs by %g: got %g want %g",
						j, k, k/got.Cols(), k%got.Cols(), diff, got.Data[k], ref.Data[k])
				}
				lastDiff = maxDiff(got.Data, ref.Data)
			}

			ref, err := want.Tensor(fmt.Sprintf("case%d.mean_pooled", i))
			if err != nil {
				t.Fatal(err)
			}
			pooled := pool.Mean(states[len(states)-1])
			cos := cosine(pooled, ref.Data)
			if cos < minCosine {
				t.Errorf("mean pooled cosine %.6f, want >= %v", cos, minCosine)
			}
			t.Logf("last hidden state max |diff| %.2g; pooled cosine %.9f", lastDiff, cos)
			if k, diff := firstMismatch(pooled, ref.Data); k >= 0 {
				t.Errorf("mean pooled dim %d differs by %g", k, diff)
			}
		})
	}
}

func TestForwardRejectsBadInput(t *testing.T) {
	model, err := Load(golden.ModelDir(t, "INFER_MINILM_DIR", "all-MiniLM-L6-v2"))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	tests := []struct {
		name string
		ids  []int
	}{
		{"empty", nil},
		{"negative id", []int{101, -1, 102}},
		{"id past vocab", []int{101, model.Config.VocabSize, 102}},
		{"too long", make([]int, model.Config.MaxPositionEmbeddings+1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := model.Forward(tt.ids); err == nil {
				t.Error("Forward succeeded, want an error")
			}
		})
	}
}

func firstMismatch(got, want []float32) (int, float64) {
	for k := range got {
		diff := math.Abs(float64(got[k]) - float64(want[k]))
		if diff > atol+rtol*math.Abs(float64(want[k])) || math.IsNaN(diff) {
			return k, diff
		}
	}
	return -1, 0
}

func maxDiff(got, want []float32) float64 {
	var m float64
	for k := range got {
		m = max(m, math.Abs(float64(got[k])-float64(want[k])))
	}
	return m
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / math.Sqrt(na*nb)
}

func BenchmarkMiniLMForward(b *testing.B) {
	for _, device := range testDevices(b) {
		b.Run(device, func(b *testing.B) { benchMiniLM(b, device) })
	}
}

func benchMiniLM(b *testing.B, device string) {
	model, err := LoadWith(golden.ModelDir(b, "INFER_MINILM_DIR", "all-MiniLM-L6-v2"), Options{Device: device})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	for _, c := range golden.Cases(b, "minilm_model.json") {
		b.Run(fmt.Sprintf("%d_tokens", len(c.IDs)), func(b *testing.B) {
			for b.Loop() {
				if _, err := model.Forward(c.IDs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
