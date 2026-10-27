package llama

import (
	"slices"
	"testing"

	"github.com/mhpenta/infer/sample"
)

// generateWith collects GenerateWith's tokens.
func generateWith(t *testing.T, m *Model, ids []int, maxNew int, s *sample.Sampler) []int {
	t.Helper()
	var out []int
	if err := m.GenerateWith(ids, maxNew, s, func(token int, _ []float32) bool {
		out = append(out, token)
		return true
	}); err != nil {
		t.Fatal(err)
	}
	return out
}

// TestGenerateWithSampler checks that samplers which can only choose the
// most likely token reproduce PyTorch's greedy tokens, and that sampling is
// reproducible for a seed and varies across seeds.
func TestGenerateWithSampler(t *testing.T) {
	m, cases, _ := loadMiniCPM(t)
	c := cases[2] // 65-token chat prompt, 34 tokens out
	for _, cfg := range []sample.Config{{}, {Temperature: 1, TopK: 1}} {
		s, err := sample.New(cfg, 1)
		if err != nil {
			t.Fatal(err)
		}
		if got := generateWith(t, m, c.InputIDs, len(c.OutputIDs), s); !slices.Equal(got, c.OutputIDs) {
			t.Errorf("%+v generated %v\nwant %v", cfg, got, c.OutputIDs)
		}
	}
	draw := func(seed uint64) []int {
		s, err := sample.New(sample.Config{Temperature: 1.2, TopK: 50, TopP: 0.95}, seed)
		if err != nil {
			t.Fatal(err)
		}
		return generateWith(t, m, c.InputIDs, 16, s)
	}
	a, b := draw(7), draw(7)
	if !slices.Equal(a, b) {
		t.Errorf("seed 7 gave %v, then %v", a, b)
	}
	if other := draw(8); slices.Equal(a, other) {
		t.Errorf("seeds 7 and 8 both gave %v", a)
	}
}
