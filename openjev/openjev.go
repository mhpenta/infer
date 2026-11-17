// Package openjev makes typed decisions with OpenJev's NLI cross-encoders
// (huggingface.co/AlexWortega/openjev): Qwen3.5 checkpoints that read a
// premise and a hypothesis and classify the pair as contradiction,
// entailment or neutral, run by package qwen35.
//
// A decision follows OpenJev's own code/openjev_decide.py. Each option of a
// question becomes a hypothesis about the state,
//
//	The answer to "{instructions}" is {option}: {rubric}
//
// with the option itself as its rubric when none is given, and the option's
// probability is its P(entailment), normalized over the options. Nothing is
// generated, so the answer is always one of the options. Choice, score and
// yes/no questions are all scored this way; they differ only in how the
// caller reads the distribution.
//
// All of a state's options, across questions, share the text up to the
// hypothesis, so the model runs that prefix once (qwen35's ClassifyShared)
// and each option only from where its tokens differ.
package openjev

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/mhpenta/infer/model/qwen35"
	"github.com/mhpenta/infer/tokenizer/bpe"
)

// defaultTemplate is OpenJev's NLI input, when config.json has no
// nli_template.
const defaultTemplate = "Premise: {premise}\nHypothesis: {hypothesis}"

// hypothesisTemplate turns an option into a hypothesis, as openjev_decide.py
// does.
const hypothesisTemplate = `The answer to "{instructions}" is {option}: {rubric}`

// WindowChars is the longest state scored in one piece. A longer state is
// scored in windows of this many characters that overlap by 2,000, and an
// option's P(entailment) is its highest over the windows: a claim one window
// supports is supported by the state. This is openjev_decide.py's window.
const WindowChars = 24_000

const windowOverlap = 2_000

// Model is an OpenJev cross-encoder and its tokenizer. It is safe for
// concurrent use.
type Model struct {
	model      *qwen35.Model
	tok        *bpe.Tokenizer
	template   string
	entailment int // the entailment label's index
}

// Load loads the checkpoint in dir: config.json, the weights and
// tokenizer.json.
func Load(dir string) (*Model, error) { return LoadWith(dir, qwen35.Options{}) }

// LoadWith is Load with the model's options.
func LoadWith(dir string, opts qwen35.Options) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("openjev: %w", err)
	}
	var config struct {
		NLITemplate string `json:"nli_template"`
	}
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("openjev: parse config.json: %w", err)
	}
	if config.NLITemplate == "" {
		config.NLITemplate = defaultTemplate
	}
	if !strings.Contains(config.NLITemplate, "{premise}") || !strings.Contains(config.NLITemplate, "{hypothesis}") {
		return nil, fmt.Errorf("openjev: nli_template %q lacks {premise} or {hypothesis}", config.NLITemplate)
	}
	model, err := qwen35.LoadWith(dir, opts)
	if err != nil {
		return nil, err
	}
	entailment := slices.Index(model.Config.Labels, "entailment")
	if entailment < 0 {
		model.Close()
		return nil, fmt.Errorf("openjev: the classifier's labels %q have no entailment", model.Config.Labels)
	}
	tok, err := bpe.LoadDir(dir)
	if err != nil {
		model.Close()
		return nil, err
	}
	return &Model{model: model, tok: tok, template: config.NLITemplate, entailment: entailment}, nil
}

// Close releases the weight files.
func (m *Model) Close() error { return m.model.Close() }

// Labels returns the classifier's labels, in the order NLI returns them.
func (m *Model) Labels() []string { return slices.Clone(m.model.Config.Labels) }

// Text returns the model's input for a premise and a hypothesis, each with
// surrounding whitespace removed.
func (m *Model) Text(premise, hypothesis string) string {
	return strings.NewReplacer("{premise}", strings.TrimSpace(premise), "{hypothesis}", strings.TrimSpace(hypothesis)).Replace(m.template)
}

// NLI returns the probability of each label (Labels) for the pair.
func (m *Model) NLI(premise, hypothesis string) ([]float64, error) {
	logits, err := m.model.Classify(m.tok.Encode(m.Text(premise, hypothesis)))
	if err != nil {
		return nil, err
	}
	return softmax(logits), nil
}

// Question is one typed decision about a state.
type Question struct {
	// Instructions is the question, such as "Is the refund permitted?".
	Instructions string
	// Options are the possible answers, such as "no" and "yes".
	Options []string
	// Rubric, if set, says what each option means; an option it does not
	// cover stands for itself.
	Rubric map[string]string
}

// Answer is a question's probability for each option, in the question's
// order, summing to 1.
type Answer []float64

// Best returns the index of the most probable option, the first on ties.
func (a Answer) Best() int {
	best := 0
	for i, p := range a {
		if p > a[best] {
			best = i
		}
	}
	return best
}

// Hypothesis returns the hypothesis option i of q stands for.
func (q Question) Hypothesis(i int) string {
	o := q.Options[i]
	rubric, ok := q.Rubric[o]
	if !ok {
		rubric = o
	}
	return strings.NewReplacer("{instructions}", strings.TrimSpace(q.Instructions), "{option}", o, "{rubric}", rubric).Replace(hypothesisTemplate)
}

// Decide answers each question about state.
func (m *Model) Decide(state string, questions []Question) ([]Answer, error) {
	if len(questions) == 0 {
		return nil, nil
	}
	var hypotheses []string
	for qi, q := range questions {
		if len(q.Options) == 0 {
			return nil, fmt.Errorf("openjev: question %d has no options", qi)
		}
		for i := range q.Options {
			hypotheses = append(hypotheses, q.Hypothesis(i))
		}
	}
	// best[j] is hypothesis j's highest P(entailment) over the windows.
	best := make([]float64, len(hypotheses))
	for _, window := range windows(state) {
		seqs := make([][]int, len(hypotheses))
		for j, h := range hypotheses {
			seqs[j] = m.tok.Encode(m.Text(window, h))
		}
		// The tokenizer can merge characters across the premise's end, so
		// the shared prefix is found in the ids, not the text.
		n := commonPrefix(seqs)
		suffixes := make([][]int, len(seqs))
		for j, s := range seqs {
			suffixes[j] = s[n:]
		}
		logits, err := m.model.ClassifyShared(seqs[0][:n], suffixes)
		if err != nil {
			return nil, err
		}
		for j, l := range logits {
			best[j] = max(best[j], softmax(l)[m.entailment])
		}
	}
	answers := make([]Answer, len(questions))
	j := 0
	for qi, q := range questions {
		p := best[j : j+len(q.Options)]
		j += len(q.Options)
		var sum float64
		for _, x := range p {
			sum += x
		}
		sum = max(sum, 1e-9)
		answers[qi] = make(Answer, len(p))
		for i, x := range p {
			answers[qi][i] = x / sum
		}
	}
	return answers, nil
}

// windows splits a state longer than WindowChars characters into
// overlapping windows.
func windows(state string) []string {
	r := []rune(state)
	if len(r) <= WindowChars {
		return []string{state}
	}
	var out []string
	for start := 0; start < max(len(r)-windowOverlap, 1); start += WindowChars - windowOverlap {
		out = append(out, string(r[start:min(start+WindowChars, len(r))]))
	}
	return out
}

// commonPrefix returns how many leading ids every sequence shares, leaving
// each at least one of its own.
func commonPrefix(seqs [][]int) int {
	n := len(seqs[0]) - 1
	for _, s := range seqs[1:] {
		n = min(n, len(s)-1)
		for j := range n {
			if s[j] != seqs[0][j] {
				n = j
				break
			}
		}
	}
	return max(n, 0)
}

// softmax returns the probabilities of logits, computed in float64.
func softmax(logits []float32) []float64 {
	peak := float64(slices.Max(logits))
	p := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		p[i] = math.Exp(float64(l) - peak)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}
