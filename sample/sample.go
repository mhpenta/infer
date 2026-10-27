// Package sample chooses the next token from a model's logits: greedily, or
// by sampling with temperature, top-k, top-p (nucleus) and a repetition
// penalty, each with the meaning transformers gives it and applied in
// model.generate's order (penalty, temperature, top-k, top-p). The
// distributions it samples from are checked against transformers' logits
// processors.
package sample

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
)

// Config is how tokens are chosen. The zero value is greedy decoding.
type Config struct {
	// Temperature divides the logits before sampling: below 1 sharpens the
	// distribution, above 1 flattens it. 0 is greedy: always the most likely
	// token.
	Temperature float64

	// TopK keeps only the TopK highest logits (and any tied with the lowest
	// of them). 0 keeps all.
	TopK int

	// TopP keeps the most likely tokens, in order, until their probability
	// reaches TopP (always at least one). 0 or 1 keeps all.
	TopP float64

	// RepetitionPenalty, above 1, makes every token already in the sequence
	// (prompt included) less likely: its logit is divided by the penalty if
	// positive and multiplied by it if negative. 0 or 1 is off. It applies
	// to greedy decoding too.
	RepetitionPenalty float64
}

// Greedy reports whether c always chooses the most likely token.
func (c Config) Greedy() bool { return c.Temperature == 0 }

func (c Config) validate() error {
	switch {
	case !(c.Temperature >= 0) || math.IsInf(c.Temperature, 0):
		return fmt.Errorf("sample: temperature %g, want >= 0", c.Temperature)
	case c.TopK < 0:
		return fmt.Errorf("sample: top-k %d, want >= 0", c.TopK)
	case !(c.TopP >= 0 && c.TopP <= 1):
		return fmt.Errorf("sample: top-p %g, want 0 to 1", c.TopP)
	case !(c.RepetitionPenalty >= 0) || math.IsInf(c.RepetitionPenalty, 0):
		return fmt.Errorf("sample: repetition penalty %g, want >= 0", c.RepetitionPenalty)
	}
	return nil
}

func (c Config) penalty() bool { return c.RepetitionPenalty != 0 && c.RepetitionPenalty != 1 }
func (c Config) topP() bool    { return c.TopP != 0 && c.TopP != 1 }

// LoadConfig reads a model's generation_config.json as model.generate would
// use it, filling unset fields with transformers' defaults: greedy unless
// do_sample is true, then temperature 1, top-k 50 and top-p 1, with no
// repetition penalty. Without the file, the result is greedy.
func LoadConfig(dir string) (Config, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if errors.Is(err, os.ErrNotExist) {
		return Config{}, nil
	}
	if err != nil {
		return Config{}, fmt.Errorf("sample: %w", err)
	}
	g := struct {
		DoSample          bool     `json:"do_sample"`
		Temperature       *float64 `json:"temperature"`
		TopK              *int     `json:"top_k"`
		TopP              *float64 `json:"top_p"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}{}
	if err := json.Unmarshal(raw, &g); err != nil {
		return Config{}, fmt.Errorf("sample: parse generation_config.json: %w", err)
	}
	or := func(p *float64, def float64) float64 {
		if p == nil {
			return def
		}
		return *p
	}
	c := Config{RepetitionPenalty: or(g.RepetitionPenalty, 1)}
	if g.DoSample {
		c.Temperature, c.TopP, c.TopK = or(g.Temperature, 1), or(g.TopP, 1), 50
		if g.TopK != nil {
			c.TopK = *g.TopK
		}
	}
	if err := c.validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}

// Sampler chooses tokens with one Config and its own random source. It is
// not safe for concurrent use; give each generation its own.
type Sampler struct {
	c   Config
	rng *rand.Rand

	// Scratch reused across calls.
	scores []float32
	seen   []bool
	tokens []int
	probs  []float64
}

// New returns a sampler for c whose draws are determined by seed.
func New(c Config, seed uint64) (*Sampler, error) {
	if err := c.validate(); err != nil {
		return nil, err
	}
	return &Sampler{c: c, rng: rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))}, nil
}

// Config returns the sampler's settings.
func (s *Sampler) Config() Config { return s.c }

// Next chooses the token to follow seq (the prompt and every token generated
// so far, which the repetition penalty looks at) from the model's logits for
// it. logits is not modified.
func (s *Sampler) Next(logits []float32, seq []int) int {
	tokens, probs := s.distribution(logits, seq)
	if len(tokens) == 1 {
		return tokens[0]
	}
	u := s.rng.Float64()
	cum := 0.0
	for i, p := range probs {
		if cum += p; u < cum {
			return tokens[i]
		}
	}
	return tokens[len(tokens)-1] // u beyond a sum rounded below 1
}

// distribution returns the tokens Next can choose for these logits and their
// probabilities, which sum to 1: one token when greedy. When top-p is active
// the tokens come most likely first; otherwise in vocabulary order. The
// slices are scratch, valid until the next call.
func (s *Sampler) distribution(logits []float32, seq []int) ([]int, []float64) {
	scores := append(s.scores[:0], logits...)
	s.scores = scores
	if s.c.penalty() {
		s.penalize(scores, seq)
	}
	if s.c.Greedy() {
		best := 0
		for i, v := range scores {
			if v > scores[best] {
				best = i
			}
		}
		s.tokens, s.probs = append(s.tokens[:0], best), append(s.probs[:0], 1)
		return s.tokens, s.probs
	}

	// Temperature and top-k decide which tokens remain and their scores.
	// The division is in float32, as transformers does it on float32 logits.
	if t := float32(s.c.Temperature); t != 1 {
		for i := range scores {
			scores[i] /= t
		}
	}
	floor := float32(math.Inf(-1))
	if k := s.c.TopK; k > 0 && k < len(scores) {
		floor = kthLargest(scores, k)
	}
	tokens, probs := s.tokens[:0], s.probs[:0]
	peak := float32(math.Inf(-1))
	for _, v := range scores {
		if v >= floor {
			peak = max(peak, v)
		}
	}
	total := 0.0
	for i, v := range scores {
		if v >= floor && !math.IsInf(float64(v), -1) {
			p := math.Exp(float64(v - peak))
			tokens, probs = append(tokens, i), append(probs, p)
			total += p
		}
	}
	for i := range probs {
		probs[i] /= total
	}
	if s.c.topP() {
		tokens, probs = nucleus(tokens, probs, s.c.TopP)
	}
	s.tokens, s.probs = tokens, probs
	return tokens, probs
}

// penalize applies the repetition penalty once to each distinct token of
// seq, as transformers' RepetitionPenaltyLogitsProcessor does.
func (s *Sampler) penalize(scores []float32, seq []int) {
	if len(s.seen) < len(scores) {
		s.seen = make([]bool, len(scores))
	}
	p := float32(s.c.RepetitionPenalty)
	for _, t := range seq {
		if t < 0 || t >= len(scores) || s.seen[t] {
			continue
		}
		s.seen[t] = true
		if v := scores[t]; v < 0 {
			scores[t] = v * p
		} else {
			scores[t] = v / p
		}
	}
	for _, t := range seq {
		if t >= 0 && t < len(scores) {
			s.seen[t] = false
		}
	}
}

// kthLargest returns the k-th largest value of x (1 <= k < len(x)), keeping
// the k largest seen so far sorted in a small buffer; most values fall below
// its minimum and cost one comparison.
func kthLargest(x []float32, k int) float32 {
	top := make([]float32, 0, k) // descending
	for _, v := range x {
		if len(top) == k && v <= top[k-1] {
			continue
		}
		i, _ := slices.BinarySearchFunc(top, v, func(a, b float32) int {
			if a > b {
				return -1
			}
			return 1
		})
		if len(top) < k {
			top = append(top, 0)
		}
		copy(top[i+1:], top[i:len(top)-1])
		top[i] = v
	}
	return top[k-1]
}

// nucleus keeps the tokens transformers' TopPLogitsWarper keeps: in order of
// probability, a token stays while the probability of the tokens strictly
// above it is below topP (so the token that reaches topP stays, and the most
// likely always does). It renormalizes and returns them most likely first.
//
// Sorting the whole vocabulary would dominate a decoding step, so it sorts
// only candidates: the tokens with probability at least some floor whose
// total reaches topP. Any token below the floor has at least that total
// above it and is dropped, so the result is the same.
func nucleus(tokens []int, probs []float64, topP float64) ([]int, []float64) {
	var candidates []int
	for floor := slices.Max(probs) / 1024; ; floor /= 1024 { // floor reaches 0, taking every token
		candidates = candidates[:0]
		mass := 0.0
		for i, p := range probs {
			if p >= floor {
				candidates = append(candidates, i)
				mass += p
			}
		}
		if mass >= topP || floor == 0 {
			break
		}
	}
	slices.SortFunc(candidates, func(a, b int) int {
		if probs[a] != probs[b] {
			if probs[a] > probs[b] {
				return -1
			}
			return 1
		}
		return tokens[a] - tokens[b]
	})
	keptTokens, keptProbs := make([]int, 0, len(candidates)), make([]float64, 0, len(candidates))
	above := 0.0
	for _, c := range candidates {
		if len(keptTokens) > 0 && above >= topP {
			break
		}
		keptTokens, keptProbs = append(keptTokens, tokens[c]), append(keptProbs, probs[c])
		above += probs[c]
	}
	for i := range keptProbs {
		keptProbs[i] /= above
	}
	return keptTokens, keptProbs
}
