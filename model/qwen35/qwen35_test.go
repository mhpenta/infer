package qwen35

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// longCase is the 2,575-token premise, skipped with -short.
const longCase = 3

// relTol is the largest difference from PyTorch allowed in a hidden state or
// the logits, relative to that tensor's largest magnitude. The linear layers
// run the delta rule one position at a time, where PyTorch runs it in chunks
// of 64; the difference stays under 5e-6 through 2,575 positions.
const relTol = 1e-5

type nliCase struct {
	Premise    string    `json:"premise"`
	Hypothesis string    `json:"hypothesis"`
	Text       string    `json:"text"`
	InputIDs   []int     `json:"input_ids"`
	Probs      []float64 `json:"probs"`
}

type decisionCase struct {
	StateText    string `json:"state_text"`
	Instructions string `json:"instructions"`
	Options      []struct {
		Option     string  `json:"option"`
		Hypothesis string  `json:"hypothesis"`
		Text       string  `json:"text"`
		InputIDs   []int   `json:"input_ids"`
		P          float64 `json:"p"`
	} `json:"options"`
}

type goldens struct {
	Labels    map[string]string `json:"labels"`
	Cases     []nliCase         `json:"cases"`
	Decisions []decisionCase    `json:"decisions"`
}

// goldenModel is a checkpoint with PyTorch goldens: its directory under
// testdata/models (or envVar), the goldens' file prefix, the cases with
// hidden states, and its shape: layers, value heads per key head in the
// linear layers.
type goldenModel struct {
	envVar, dir, prefix string
	hiddenCases         []int
	layers, valuePerKey int
	large               bool // skipped with -short
}

var goldenModels = []goldenModel{
	{"INFER_OPENJEV_DIR", "openjev-0.8b", "openjev_0.8b", []int{0, 2}, 24, 1, false},
	// The 4B (9.1 GB, not in CI: scripts/fetch_models.sh openjev-4b) shares
	// the 0.8B's tokenizer, so its goldens are the model's alone, with
	// hidden states for case 0 only. Its linear layers have two value heads
	// per key head.
	{"INFER_OPENJEV_4B_DIR", "openjev-4b", "openjev_4b", []int{0}, 32, 2, true},
}

// loaded is a model with its goldens.
type loaded struct {
	model *Model
	gold  goldens
	want  *safetensors.File
}

var shared = map[string]*loaded{}

// load loads a checkpoint and its goldens once for all tests.
func (g goldenModel) load(t testing.TB) (*Model, goldens, *safetensors.File) {
	t.Helper()
	if g.large && testing.Short() {
		t.Skip("large model skipped with -short")
	}
	l := shared[g.dir]
	if l == nil {
		m, err := Load(golden.ModelDir(t, g.envVar, g.dir))
		if err != nil {
			t.Fatal(err)
		}
		l = &loaded{model: m}
		golden.JSON(t, g.prefix+"_model.json", &l.gold)
		if l.want, err = safetensors.Open(golden.Path(t, g.prefix+"_model.safetensors")); err != nil {
			t.Fatal(err)
		}
		shared[g.dir] = l
	}
	return l.model, l.gold, l.want
}

// load loads the 0.8B, for the tests that need one checkpoint.
func load(t testing.TB) (*Model, goldens, *safetensors.File) { return goldenModels[0].load(t) }

// forEachModel runs fn as a subtest for every checkpoint.
func forEachModel(t *testing.T, fn func(t *testing.T, g goldenModel, m *Model, gold goldens, want *safetensors.File)) {
	for _, g := range goldenModels {
		t.Run(g.dir, func(t *testing.T) {
			m, gold, want := g.load(t)
			fn(t, g, m, gold, want)
		})
	}
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

func TestConfig(t *testing.T) {
	forEachModel(t, func(t *testing.T, g goldenModel, m *Model, gold goldens, want *safetensors.File) {
		c := m.Config
		full := 0
		for _, l := range m.layers {
			if l.full {
				full++
			}
		}
		if c.NumHiddenLayers != g.layers || full != g.layers/4 || c.rotaryDim() != 64 ||
			c.LinearNumValueHeads != g.valuePerKey*c.LinearNumKeyHeads {
			t.Errorf("layers %d (%d full), rotary %d, %d linear value heads over %d key heads; want %d (%d full), 64, %d per key head",
				c.NumHiddenLayers, full, c.rotaryDim(), c.LinearNumValueHeads, c.LinearNumKeyHeads, g.layers, g.layers/4, g.valuePerKey)
		}
		for i, label := range c.Labels {
			if gold.Labels[fmt.Sprint(i)] != label {
				t.Errorf("label %d is %q, want %q", i, label, gold.Labels[fmt.Sprint(i)])
			}
		}
		if got, w := m.rope.InvFreq(), wantTensor(t, want, "rope.inv_freq").Data; !slices.Equal(got, w) {
			t.Errorf("inverse frequencies\n got %v\nwant %v", got, w)
		}
	})
}

func TestHiddenStatesMatchPython(t *testing.T) {
	forEachModel(t, func(t *testing.T, g goldenModel, m *Model, gold goldens, want *safetensors.File) {
		for _, i := range g.hiddenCases {
			states, err := m.HiddenStates(gold.Cases[i].InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			for j, s := range states {
				w := wantTensor(t, want, fmt.Sprintf("case%d.hidden.%d", i, j))
				if d := relDiff(s.Data, w.Data); d > relTol {
					t.Errorf("case %d, hidden state %d: %.2g from PyTorch, relative to its range", i, j, d)
				}
			}
		}
	})
}

func TestClassifyMatchesPython(t *testing.T) {
	forEachModel(t, func(t *testing.T, g goldenModel, m *Model, gold goldens, want *safetensors.File) {
		for i, c := range gold.Cases {
			if i == longCase && testing.Short() {
				continue
			}
			logits, err := m.Classify(c.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			w := wantTensor(t, want, fmt.Sprintf("case%d.logits", i)).Data
			if d := relDiff(logits, w); d > relTol {
				t.Errorf("case %d (%d tokens): logits %v, want %v (%.2g)", i, len(c.InputIDs), logits, w, d)
			}
		}
	})
}

// TestClassifySharedMatchesClassify scores each decision's options from
// their shared token prefix: the logits must be bit-identical to running
// each option alone, and match PyTorch's.
func TestClassifySharedMatchesClassify(t *testing.T) {
	forEachModel(t, func(t *testing.T, g goldenModel, m *Model, gold goldens, want *safetensors.File) {
		checkDecisions(t, m, gold, want)
	})
}

func checkDecisions(t *testing.T, m *Model, gold goldens, want *safetensors.File) {
	for i, d := range gold.Decisions {
		var ids [][]int
		for _, o := range d.Options {
			ids = append(ids, o.InputIDs)
		}
		n := commonPrefix(ids)
		var suffixes [][]int
		for _, s := range ids {
			suffixes = append(suffixes, s[n:])
		}
		got, err := m.ClassifyShared(ids[0][:n], suffixes)
		if err != nil {
			t.Fatal(err)
		}
		w := wantTensor(t, want, fmt.Sprintf("decision%d.logits", i))
		for j, s := range ids {
			alone, err := m.Classify(s)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got[j], alone) {
				t.Errorf("decision %d, option %d: shared prefix of %d tokens gives %v, alone %v", i, j, n, got[j], alone)
			}
			if dd := relDiff(got[j], w.Row(j)); dd > relTol {
				t.Errorf("decision %d, option %d: logits %v, want %v (%.2g)", i, j, got[j], w.Row(j), dd)
			}
		}
	}
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
	return n
}

func TestClassifyShared(t *testing.T) {
	m, g, _ := load(t)
	ids := g.Cases[0].InputIDs
	alone, err := m.Classify(ids)
	if err != nil {
		t.Fatal(err)
	}
	// No prefix at all, and a suffix of one token.
	for _, cut := range []int{0, len(ids) - 1} {
		got, err := m.ClassifyShared(ids[:cut], [][]int{ids[cut:]})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got[0], alone) {
			t.Errorf("prefix of %d: %v, want %v", cut, got[0], alone)
		}
	}
	if _, err := m.ClassifyShared(ids, [][]int{{}}); err == nil {
		t.Error("an empty suffix was accepted")
	}
}

func TestBadIDs(t *testing.T) {
	m, _, _ := load(t)
	for _, ids := range [][]int{nil, {-1}, {m.Config.VocabSize}} {
		if _, err := m.Classify(ids); err == nil {
			t.Errorf("Classify(%v) succeeded", ids)
		}
		if _, err := m.HiddenStates(ids); err == nil {
			t.Errorf("HiddenStates(%v) succeeded", ids)
		}
	}
}

// TestInt8 checks that int8 weights move the classifier's probabilities by
// little: at most 0.0031 on these cases, against 1e-6 for bf16.
func TestInt8(t *testing.T) {
	m, g, _ := load(t)
	q, err := LoadWith(golden.ModelDir(t, "INFER_OPENJEV_DIR", "openjev-0.8b"), Options{Int8: true})
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	for i, c := range g.Cases {
		if i == longCase {
			continue
		}
		a, err := m.Classify(c.InputIDs)
		if err != nil {
			t.Fatal(err)
		}
		b, err := q.Classify(c.InputIDs)
		if err != nil {
			t.Fatal(err)
		}
		pa, pb := softmax(a), softmax(b)
		for j := range pa {
			if math.Abs(pa[j]-pb[j]) > 0.01 {
				t.Errorf("case %d: int8 probabilities %v, bf16 %v", i, pb, pa)
				break
			}
		}
	}
}

func softmax(logits []float32) []float64 {
	p := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		p[i] = math.Exp(float64(l))
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}
