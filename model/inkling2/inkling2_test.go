package inkling2

import (
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// tinyTol is the largest difference from PyTorch allowed in a hidden state of
// the tiny checkpoint, relative to that state's largest magnitude: float32
// rounding, which grows to about 5e-6 over its ten layer visits.
const tinyTol = 1e-5

// tinyLogitTol is tinyTol for the logits. The tiny config softcaps them at 5
// to exercise the cap, which squeezes their range (to ±5, from ±15 before
// it) more than their error, so the same error is larger relative to the
// range: about 2e-5.
const tinyLogitTol = 5e-5

type tinyGolden struct {
	InputIDs  []int `json:"input_ids"`
	Generated []int `json:"generated"`
}

// loadTiny loads the tiny random checkpoint that scripts/dump_golden_inkling2.py
// writes in the original layout, and its goldens.
func loadTiny(t *testing.T) (*Model, tinyGolden, *safetensors.File) {
	t.Helper()
	m, err := Load(golden.Path(t, "inkling2_tiny"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	var gold tinyGolden
	golden.JSON(t, "inkling2_tiny_model.json", &gold)
	want, err := safetensors.Open(golden.Path(t, "inkling2_tiny_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { want.Close() })
	return m, gold, want
}

func wantTensor(t testing.TB, f *safetensors.File, name string) *tensor.Tensor {
	t.Helper()
	w, err := f.Tensor(name)
	if err != nil {
		t.Fatal(err)
	}
	return w
}

// relDiff returns the largest difference between got and want relative to
// want's largest magnitude.
func relDiff(got, want []float32) float64 {
	var diff, peak float64
	for i, w := range want {
		diff = max(diff, math.Abs(float64(got[i])-float64(w)))
		peak = max(peak, math.Abs(float64(w)))
	}
	return diff / peak
}

func TestTinyHiddenStates(t *testing.T) {
	m, gold, want := loadTiny(t)
	states, err := m.HiddenStates(gold.InputIDs)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != m.Config.Visits()+1 {
		t.Fatalf("%d states, want %d", len(states), m.Config.Visits()+1)
	}
	for j, got := range states {
		w := wantTensor(t, want, "hidden."+strconv.Itoa(j))
		if d := relDiff(got.Data, w.Data); d > tinyTol || math.IsNaN(d) {
			t.Errorf("hidden state %d differs by %.3g of its range", j, d)
		}
	}
}

func TestTinyLogits(t *testing.T) {
	m, gold, want := loadTiny(t)
	got, err := m.Logits(gold.InputIDs)
	if err != nil {
		t.Fatal(err)
	}
	w := wantTensor(t, want, "logits")
	if d := relDiff(got.Data, w.Data); d > tinyLogitTol || math.IsNaN(d) {
		t.Errorf("logits differ by %.3g of their range", d)
	}
}

// TestTinyGenerate checks greedy decoding with the caches against
// transformers' generate, token by token and logit by logit.
func TestTinyGenerate(t *testing.T) {
	m, gold, want := loadTiny(t)
	wantLogits := wantTensor(t, want, "generate.logits")
	var got []int
	err := m.GenerateWith(gold.InputIDs, len(gold.Generated), nil, func(token int, logits []float32) bool {
		step := len(got)
		if d := relDiff(logits, wantLogits.Row(step)); d > tinyLogitTol || math.IsNaN(d) {
			t.Errorf("step %d: logits differ by %.3g of their range", step, d)
		}
		got = append(got, token)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(gold.Generated) {
		t.Fatalf("generated %v, want %v", got, gold.Generated)
	}
	for i := range got {
		if got[i] != gold.Generated[i] {
			t.Fatalf("generated %v, want %v", got, gold.Generated)
		}
	}
}

// TestTinyDenseExperts runs the experts as prefill does for long inputs,
// dequantizing each expert's weights for the dense products, which the
// tiny input (a few positions per expert) otherwise never reaches.
func TestTinyDenseExperts(t *testing.T) {
	m, gold, want := loadTiny(t)
	defer func(n int) { denseRows = n }(denseRows)
	denseRows = 1
	got, err := m.Logits(gold.InputIDs)
	if err != nil {
		t.Fatal(err)
	}
	if d := relDiff(got.Data, wantTensor(t, want, "logits").Data); d > tinyLogitTol || math.IsNaN(d) {
		t.Errorf("logits differ by %.3g of their range", d)
	}
}

// TestTinySlidingCacheTrim decodes far past the sliding window, which trims
// the sliding layers' caches, and checks each step's logits against running
// the whole sequence at once, whose queries all attend before any trimming.
func TestTinySlidingCacheTrim(t *testing.T) {
	m, gold, _ := loadTiny(t)
	ids := slices.Clone(gold.InputIDs)
	for len(ids) < 4*m.Config.SlidingWindowSize {
		ids = append(ids, gold.InputIDs...)
	}
	whole, err := m.Logits(ids)
	if err != nil {
		t.Fatal(err)
	}
	st := m.newState()
	logits := tensor.New(1, m.Config.UnpaddedVocabSize)
	for i, id := range ids {
		y := st.forward([]int{id}, nil)
		m.logitsInto(logits, y)
		if d := relDiff(logits.Data, whole.Row(i)); d > tinyLogitTol {
			t.Fatalf("position %d: logits differ by %.3g of their range", i, d)
		}
	}
	trimmed := 0
	for _, vs := range st.visits {
		if vs.start == 0 {
			continue
		}
		trimmed++
		if n := len(vs.keys) / (m.Config.SWANumKeyValueHeads * m.Config.SWAHeadDim); n > 2*m.Config.SlidingWindowSize {
			t.Errorf("a sliding cache holds %d positions", n)
		}
	}
	if trimmed == 0 {
		t.Error("no sliding cache was trimmed")
	}
}
