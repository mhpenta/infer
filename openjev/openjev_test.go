package openjev

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
)

// probTol bounds the difference from PyTorch's probabilities; the logits
// they come from match within 1e-5 of their range (see package qwen35).
const probTol = 1e-5

// longCase is the 2,575-token premise, skipped with -short.
const longCase = 3

type goldens struct {
	Cases []struct {
		Premise    string    `json:"premise"`
		Hypothesis string    `json:"hypothesis"`
		Text       string    `json:"text"`
		InputIDs   []int     `json:"input_ids"`
		Probs      []float64 `json:"probs"`
	} `json:"cases"`
	Decisions []struct {
		StateText    string            `json:"state_text"`
		Instructions string            `json:"instructions"`
		Rubric       map[string]string `json:"rubric"`
		Options      []struct {
			Option     string  `json:"option"`
			Hypothesis string  `json:"hypothesis"`
			Text       string  `json:"text"`
			InputIDs   []int   `json:"input_ids"`
			P          float64 `json:"p"`
		} `json:"options"`
	} `json:"decisions"`
}

var shared struct {
	model *Model
	gold  goldens
}

func load(t testing.TB) (*Model, goldens) {
	t.Helper()
	if shared.model == nil {
		m, err := Load(golden.ModelDir(t, "INFER_OPENJEV_DIR", "openjev-0.8b"))
		if err != nil {
			t.Fatal(err)
		}
		golden.JSON(t, "openjev_0.8b_model.json", &shared.gold)
		shared.model = m
	}
	return shared.model, shared.gold
}

func TestNLIMatchesPython(t *testing.T) {
	m, g := load(t)
	if got := m.Labels(); !slices.Equal(got, []string{"contradiction", "entailment", "neutral"}) {
		t.Fatalf("labels %q", got)
	}
	for i, c := range g.Cases {
		if i == longCase && testing.Short() {
			continue
		}
		if text := m.Text(c.Premise, c.Hypothesis); text != c.Text {
			t.Errorf("case %d: text %q, want %q", i, text, c.Text)
		}
		if ids := m.tok.Encode(c.Text); !slices.Equal(ids, c.InputIDs) {
			t.Errorf("case %d: ids differ from Python's", i)
		}
		probs, err := m.NLI(c.Premise, c.Hypothesis)
		if err != nil {
			t.Fatal(err)
		}
		for j, p := range probs {
			if math.Abs(p-c.Probs[j]) > probTol {
				t.Errorf("case %d: probabilities %v, want %v", i, probs, c.Probs)
				break
			}
		}
	}
}

func TestDecideMatchesPython(t *testing.T) {
	m, g := load(t)
	for i, d := range g.Decisions {
		q := Question{Instructions: d.Instructions, Rubric: d.Rubric}
		for _, o := range d.Options {
			q.Options = append(q.Options, o.Option)
		}
		for j, o := range d.Options {
			if h := q.Hypothesis(j); h != o.Hypothesis {
				t.Errorf("decision %d, option %d: hypothesis %q, want %q", i, j, h, o.Hypothesis)
			}
			if ids := m.tok.Encode(m.Text(d.StateText, o.Hypothesis)); !slices.Equal(ids, o.InputIDs) {
				t.Errorf("decision %d, option %d: ids differ from Python's", i, j)
			}
		}
		answers, err := m.Decide(d.StateText, []Question{q})
		if err != nil {
			t.Fatal(err)
		}
		for j, o := range d.Options {
			if math.Abs(answers[0][j]-o.P) > probTol {
				t.Errorf("decision %d: answer %v, option %d want %g", i, answers[0], j, o.P)
			}
		}
	}
}

// TestDecideQuestionsTogether checks that asking several questions in one
// call, which shares the state's prefix across all of them, answers each as
// asking it alone does.
func TestDecideQuestionsTogether(t *testing.T) {
	m, _ := load(t)
	state := "Order 1182: two lamps, paid by card, shipped yesterday. The customer writes: one lamp arrived broken, please send a new one."
	questions := []Question{
		{Instructions: "What does the customer want?", Options: []string{"a refund", "a replacement", "order status"}},
		{Instructions: "Does the customer report damage?", Options: []string{"no", "yes"},
			Rubric: map[string]string{"no": "Nothing is damaged.", "yes": "Something arrived damaged."}},
	}
	together, err := m.Decide(state, questions)
	if err != nil {
		t.Fatal(err)
	}
	for i, q := range questions {
		alone, err := m.Decide(state, []Question{q})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(together[i], alone[0]) {
			t.Errorf("question %d: together %v, alone %v", i, together[i], alone[0])
		}
	}
	if together[0].Best() != 1 || together[1].Best() != 1 {
		t.Errorf("answers %v: want a replacement, and yes", together)
	}
}

func TestDecideRejectsEmptyQuestion(t *testing.T) {
	m, _ := load(t)
	if _, err := m.Decide("state", []Question{{Instructions: "q"}}); err == nil {
		t.Error("a question without options was accepted")
	}
	if a, err := m.Decide("state", nil); err != nil || a != nil {
		t.Errorf("no questions: %v, %v", a, err)
	}
}

func TestWindows(t *testing.T) {
	if w := windows("short"); !slices.Equal(w, []string{"short"}) {
		t.Errorf("windows(short) = %q", w)
	}
	state := strings.Repeat("é", 50_000) // runes, not bytes, as Python slices str
	w := windows(state)
	// Starts every 22,000 characters before len−2,000: 0, 22,000 and 44,000.
	if len(w) != 3 || len([]rune(w[0])) != WindowChars || len([]rune(w[2])) != 6_000 {
		t.Errorf("%d windows of %d, …, %d characters", len(w), len([]rune(w[0])), len([]rune(w[len(w)-1])))
	}
}

func TestBest(t *testing.T) {
	if b := (Answer{0.2, 0.5, 0.5, 0.1}).Best(); b != 1 {
		t.Errorf("Best = %d, want the first of the ties", b)
	}
}
