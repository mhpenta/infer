package sample

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
)

type goldenCase struct {
	Temperature       float64   `json:"temperature"`
	TopK              int       `json:"top_k"`
	TopP              float64   `json:"top_p"`
	RepetitionPenalty float64   `json:"repetition_penalty"`
	Sequence          []int     `json:"sequence"`
	Logits            []float32 `json:"logits"`
	Tensor            string    `json:"tensor"`
	Row               int       `json:"row"`
	Tokens            []int     `json:"tokens"`
	Probs             []float64 `json:"probs"`
}

// TestMatchesTransformers checks the distribution sampled from against
// transformers' logits processors on synthetic logits and on real
// MiniCPM5-1B logits: the same tokens, with the same probabilities.
func TestMatchesTransformers(t *testing.T) {
	var cases []goldenCase
	golden.JSON(t, "sample.json", &cases)
	real, err := safetensors.Open(golden.Path("minicpm5_1b_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer real.Close()
	for i, c := range cases {
		cfg := Config{Temperature: c.Temperature, TopK: c.TopK, TopP: c.TopP, RepetitionPenalty: c.RepetitionPenalty}
		t.Run(fmt.Sprintf("case%d_%+v", i, cfg), func(t *testing.T) {
			logits := c.Logits
			if c.Tensor != "" {
				x, err := real.Tensor(c.Tensor)
				if err != nil {
					t.Fatal(err)
				}
				logits = x.Row(c.Row)
			}
			s, err := New(cfg, 1)
			if err != nil {
				t.Fatal(err)
			}
			tokens, probs := s.distribution(logits, c.Sequence)
			got := map[int]float64{}
			for j, tok := range tokens {
				got[tok] = probs[j]
			}
			if len(got) != len(c.Tokens) {
				t.Fatalf("%d tokens can be drawn, transformers keeps %d", len(got), len(c.Tokens))
			}
			for j, tok := range c.Tokens {
				p, ok := got[tok]
				if !ok {
					t.Fatalf("token %d (probability %g) is not kept", tok, c.Probs[j])
				}
				if math.Abs(p-c.Probs[j]) > 2e-6 {
					t.Errorf("token %d: probability %g, transformers %g", tok, p, c.Probs[j])
				}
			}
		})
	}
}

func TestGreedy(t *testing.T) {
	logits := []float32{1, 5, -2, 5, 4}
	s, err := New(Config{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Next(logits, nil); got != 1 {
		t.Errorf("greedy chose %d, want 1 (the first of the tied maxima)", got)
	}
	// A penalty of 2 turns 5 into 2.5 for tokens 1 and 3, leaving 4 on top.
	s, err = New(Config{RepetitionPenalty: 2}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Next(logits, []int{3, 1, 1}); got != 4 {
		t.Errorf("greedy with a penalty chose %d, want 4", got)
	}
	if logits[1] != 5 {
		t.Error("Next modified the logits")
	}
}

// TestDrawFrequencies draws many tokens from a fixed distribution and checks
// the counts with a chi-square test.
func TestDrawFrequencies(t *testing.T) {
	logits := []float32{2, 1, 0.5, 0, -1, -3, 1.5, 0.2}
	for _, cfg := range []Config{
		{Temperature: 1},
		{Temperature: 0.7, TopK: 5},
		{Temperature: 1.2, TopP: 0.8},
	} {
		s, err := New(cfg, 42)
		if err != nil {
			t.Fatal(err)
		}
		tokens, probs := s.distribution(logits, nil)
		want := map[int]float64{}
		for i, tok := range tokens {
			want[tok] = probs[i]
		}
		const draws = 200_000
		counts := map[int]int{}
		for range draws {
			counts[s.Next(logits, nil)]++
		}
		chi2 := 0.0
		for tok, n := range counts {
			p, ok := want[tok]
			if !ok {
				t.Fatalf("%+v: drew token %d, which has no probability", cfg, tok)
			}
			chi2 += math.Pow(float64(n)-p*draws, 2) / (p * draws)
		}
		// 99.9999th percentile of chi-square with 7 degrees of freedom.
		if chi2 > 40 {
			t.Errorf("%+v: chi-square %.1f over %d tokens; counts %v, probabilities %v", cfg, chi2, len(want), counts, want)
		}
	}
}

func TestSeedReproducible(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	logits := make([]float32, 1000)
	for i := range logits {
		logits[i] = float32(r.NormFloat64())
	}
	draw := func(seed uint64) []int {
		s, err := New(Config{Temperature: 1, TopP: 0.9}, seed)
		if err != nil {
			t.Fatal(err)
		}
		var out []int
		for range 50 {
			out = append(out, s.Next(logits, out))
		}
		return out
	}
	if a, b := draw(5), draw(5); !slices.Equal(a, b) {
		t.Error("the same seed gave different tokens")
	}
	if a, b := draw(5), draw(6); slices.Equal(a, b) {
		t.Error("different seeds gave the same 50 tokens")
	}
}

// TestKthLargest checks the selection against sorting, with ties.
func TestKthLargest(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for range 200 {
		n := 1 + r.IntN(300)
		x := make([]float32, n+1)
		for i := range x {
			x[i] = float32(r.IntN(40)) // many ties
		}
		k := 1 + r.IntN(n)
		sorted := slices.Clone(x)
		slices.SortFunc(sorted, func(a, b float32) int { return cmp.Compare(b, a) })
		if got := kthLargest(x, k); got != sorted[k-1] {
			t.Fatalf("kthLargest(k=%d of %d) = %g, want %g", k, len(x), got, sorted[k-1])
		}
	}
}

// TestNucleusMatchesBruteForce checks the candidate shortcut against sorting
// every token, on peaked, flat and tied distributions.
func TestNucleusMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	for trial := range 300 {
		n := 1 + r.IntN(5000)
		probs := make([]float64, n)
		total := 0.0
		for i := range probs {
			switch trial % 3 {
			case 0:
				probs[i] = math.Exp(4 * r.NormFloat64()) // peaked
			case 1:
				probs[i] = 1 + r.Float64()*1e-3 // flat: candidates need several rounds
			default:
				probs[i] = float64(1 + r.IntN(3)) // ties
			}
			total += probs[i]
		}
		for i := range probs {
			probs[i] /= total
		}
		tokens := make([]int, n)
		for i := range tokens {
			tokens[i] = i * 3
		}
		topP := r.Float64()
		gotT, gotP := nucleus(slices.Clone(tokens), slices.Clone(probs), topP)

		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		slices.SortFunc(order, func(a, b int) int {
			if c := cmp.Compare(probs[b], probs[a]); c != 0 {
				return c
			}
			return cmp.Compare(a, b)
		})
		var wantT []int
		above := 0.0
		for _, i := range order {
			if len(wantT) > 0 && above >= topP {
				break
			}
			wantT = append(wantT, tokens[i])
			above += probs[i]
		}
		if !slices.Equal(gotT, wantT) {
			t.Fatalf("trial %d (n=%d, top-p %.3f): kept %d tokens, want %d", trial, n, topP, len(gotT), len(wantT))
		}
		sum := 0.0
		for _, p := range gotP {
			sum += p
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("trial %d: kept probabilities sum to %g", trial, sum)
		}
	}
}

func TestLoadConfig(t *testing.T) {
	dir := golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b")
	c, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if want := (Config{Temperature: 0.9, TopK: 50, TopP: 0.95, RepetitionPenalty: 1}); c != want {
		t.Errorf("MiniCPM5-1B: %+v, want %+v (top-k from transformers' default)", c, want)
	}
	for _, tc := range []struct {
		file string
		want Config
	}{
		{"", Config{}},
		{`{"do_sample": false, "temperature": 0.7, "repetition_penalty": 1.2}`, Config{RepetitionPenalty: 1.2}},
		{`{"do_sample": true}`, Config{Temperature: 1, TopK: 50, TopP: 1, RepetitionPenalty: 1}},
		{`{"do_sample": true, "top_k": 0, "temperature": 0.6}`, Config{Temperature: 0.6, TopP: 1, RepetitionPenalty: 1}},
	} {
		dir := t.TempDir()
		if tc.file != "" {
			if err := os.WriteFile(filepath.Join(dir, "generation_config.json"), []byte(tc.file), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		c, err := LoadConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		if c != tc.want {
			t.Errorf("%q: %+v, want %+v", tc.file, c, tc.want)
		}
	}
}

func TestNewRejectsBadConfig(t *testing.T) {
	for _, c := range []Config{
		{Temperature: -1},
		{Temperature: math.NaN()},
		{TopK: -1},
		{TopP: 1.5},
		{TopP: -0.1},
		{RepetitionPenalty: -1},
	} {
		if _, err := New(c, 1); err == nil {
			t.Errorf("New(%+v) succeeded", c)
		}
	}
}

// BenchmarkNext times one draw from real MiniCPM5-1B logits (130,560
// tokens, of which top-p 0.95 at temperature 1 keeps 598).
func BenchmarkNext(b *testing.B) {
	real, err := safetensors.Open(golden.Path("minicpm5_1b_model.safetensors"))
	if err != nil {
		b.Fatal(err)
	}
	defer real.Close()
	x, err := real.Tensor("case1.logits")
	if err != nil {
		b.Fatal(err)
	}
	logits := x.Row(0)
	r := rand.New(rand.NewPCG(7, 8))
	seq := make([]int, 500)
	for i := range seq {
		seq[i] = r.IntN(len(logits))
	}
	for _, c := range []Config{
		{},
		{Temperature: 1, TopK: 50, TopP: 0.95},
		{Temperature: 0.9, TopP: 0.95, RepetitionPenalty: 1.1},
		{Temperature: 1},
	} {
		b.Run(fmt.Sprintf("%+v", c), func(b *testing.B) {
			s, err := New(c, 1)
			if err != nil {
				b.Fatal(err)
			}
			for b.Loop() {
				s.Next(logits, seq)
			}
		})
	}
}
