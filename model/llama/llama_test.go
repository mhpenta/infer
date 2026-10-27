package llama

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
)

// relTol is the largest difference from PyTorch allowed in a hidden state or
// a row of logits, relative to that tensor's (or row's) largest value.
const relTol = 2e-5

// longCase is the 2,461-token prompt, skipped with -short.
const longCase = 3

type modelCase struct {
	Kind      string `json:"kind"`
	Text      string `json:"text"`
	InputIDs  []int  `json:"input_ids"`
	OutputIDs []int  `json:"output_ids"`
	Output    string `json:"output"`
}

var shared struct {
	model *Model
	int8  *Model
	cases []modelCase
	want  *safetensors.File
}

// loadMiniCPM loads MiniCPM5-1B and its goldens once for all tests.
func loadMiniCPM(t testing.TB) (*Model, []modelCase, *safetensors.File) {
	t.Helper()
	if shared.model == nil {
		model, err := Load(golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b"))
		if err != nil {
			t.Fatal(err)
		}
		var g struct {
			EOS   []int       `json:"eos_token_ids"`
			Cases []modelCase `json:"cases"`
		}
		golden.JSON(t, "minicpm5_1b_model.json", &g)
		if !slices.Equal(model.Config.EOS, g.EOS) {
			t.Fatalf("end tokens %v, want %v", model.Config.EOS, g.EOS)
		}
		want, err := safetensors.Open(golden.Path("minicpm5_1b_model.safetensors"))
		if err != nil {
			t.Fatal(err)
		}
		shared.model, shared.cases, shared.want = model, g.Cases, want
	}
	return shared.model, shared.cases, shared.want
}

// sequence is a case's prompt and generated tokens but the last: the input
// of the teacher-forced pass the goldens come from.
func (c modelCase) sequence() []int {
	return append(slices.Clone(c.InputIDs), c.OutputIDs[:len(c.OutputIDs)-1]...)
}

func TestConfig(t *testing.T) {
	m, _, _ := loadMiniCPM(t)
	c := m.Config
	if c.HeadDim != 128 || c.NumKeyValueHeads != 2 || c.NumAttentionHeads != 16 || c.TieWordEmbeddings {
		t.Errorf("config %+v", c)
	}
	if !m.lmHead.BF16() || !m.layers[0].qkv.BF16() || m.embed.bf16 == nil {
		t.Error("bf16 weights are not kept in bf16")
	}
}

// TestHiddenStatesMatchPython compares every layer's input and the final
// output with transformers' hidden states, and the full logits of the
// positions where generation starts.
func TestHiddenStatesMatchPython(t *testing.T) {
	m, cases, want := loadMiniCPM(t)
	for i := range 2 {
		c := cases[i]
		t.Run(fmt.Sprintf("case%d", i), func(t *testing.T) {
			states, err := m.HiddenStates(c.sequence())
			if err != nil {
				t.Fatal(err)
			}
			if len(states) != m.Config.NumHiddenLayers+1 {
				t.Fatalf("%d states, want %d", len(states), m.Config.NumHiddenLayers+1)
			}
			worst := 0.0
			for j, s := range states {
				worst = max(worst, compare(t, want, fmt.Sprintf("case%d.hidden.%d", i, j), s.Data))
			}
			ref, err := want.Tensor(fmt.Sprintf("case%d.logits", i))
			if err != nil {
				t.Fatal(err)
			}
			n := len(c.InputIDs)
			final := states[len(states)-1]
			y := tensor.FromData(final.Data[(n-1)*final.Cols():(n-1+ref.Rows())*final.Cols()], ref.Rows(), final.Cols())
			logits := tensor.New(ref.Rows(), m.Config.VocabSize)
			m.LogitsInto(logits, y)
			for r := range ref.Rows() {
				worst = max(worst, compareData(t, fmt.Sprintf("logits row %d", r), logits.Row(r), ref.Row(r)))
			}
			t.Logf("largest difference from PyTorch: %.2g of a tensor's range", worst)
		})
	}
}

// TestTopLogitsMatchPython checks the logits at PyTorch's 64 highest-scoring
// tokens, and the argmax, at every position of every case (for the long
// case, every 64th prompt position and all generated ones).
func TestTopLogitsMatchPython(t *testing.T) {
	m, cases, want := loadMiniCPM(t)
	for i, c := range cases {
		t.Run(fmt.Sprintf("case%d_%d_in", i, len(c.InputIDs)), func(t *testing.T) {
			if i == longCase && testing.Short() {
				t.Skip("long prompt skipped with -short")
			}
			seq := c.sequence()
			y := m.newState(len(seq)).forward(seq, nil)
			values, indices := topK(t, want, i, len(seq))
			var positions []int
			for p := range seq {
				if i != longCase || p%64 == 0 || p >= len(c.InputIDs)-1 {
					positions = append(positions, p)
				}
			}
			logits := tensor.New(1, m.Config.VocabSize)
			worst := 0.0
			for _, p := range positions {
				m.LogitsInto(logits, tensor.FromData(y.Row(p), 1, y.Cols()))
				worst = max(worst, checkTopK(t, p, logits.Data, values[p], indices[p]))
			}
			t.Logf("%d positions; largest difference: %.2g of a row's range", len(positions), worst)
		})
	}
}

// TestGenerateMatchesPython decodes greedily with the KV cache and checks
// every token against PyTorch's generate, and every step's logits against
// the teacher-forced ones at the same position.
func TestGenerateMatchesPython(t *testing.T) {
	m, cases, want := loadMiniCPM(t)
	for i, c := range cases {
		t.Run(fmt.Sprintf("case%d_%d_in_%d_out", i, len(c.InputIDs), len(c.OutputIDs)), func(t *testing.T) {
			if i == longCase && testing.Short() {
				t.Skip("long prompt skipped with -short")
			}
			values, indices := topK(t, want, i, len(c.InputIDs)+len(c.OutputIDs)-1)
			var got []int
			worst := 0.0
			err := m.GenerateFunc(c.InputIDs, len(c.OutputIDs), func(token int, logits []float32) bool {
				p := len(c.InputIDs) - 1 + len(got)
				worst = max(worst, checkTopK(t, p, logits, values[p], indices[p]))
				got = append(got, token)
				return true
			})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, c.OutputIDs) {
				t.Fatalf("generated %v\nwant %v", got, c.OutputIDs)
			}
			t.Logf("largest step difference: %.2g of a row's range", worst)
		})
	}
}

func TestGenerateRejectsBadInput(t *testing.T) {
	m, _, _ := loadMiniCPM(t)
	for _, ids := range [][]int{nil, {-1}, {m.Config.VocabSize}} {
		if _, err := m.Generate(ids, 1); err == nil {
			t.Errorf("Generate(%v) succeeded", ids)
		}
	}
}

// topK reads a case's golden top-k logits and token ids, one row per
// position.
func topK(t *testing.T, want *safetensors.File, i, positions int) ([][]float32, [][]int) {
	t.Helper()
	v, err := want.Tensor(fmt.Sprintf("case%d.topk.values", i))
	if err != nil {
		t.Fatal(err)
	}
	idx, shape, err := want.Ints(fmt.Sprintf("case%d.topk.indices", i))
	if err != nil {
		t.Fatal(err)
	}
	if v.Rows() != positions || shape[0] != positions {
		t.Fatalf("golden top-k has %d positions, want %d", v.Rows(), positions)
	}
	k := v.Cols()
	values, indices := make([][]float32, positions), make([][]int, positions)
	for p := range positions {
		values[p], indices[p] = v.Row(p), idx[p*k:(p+1)*k]
	}
	return values, indices
}

// checkTopK compares logits at the golden top-k tokens with their golden
// values, and requires the same argmax unless PyTorch's top two are within
// the tolerance of each other. It returns the largest difference relative
// to the row's largest golden value.
func checkTopK(t *testing.T, p int, logits, values []float32, indices []int) float64 {
	t.Helper()
	scale := 0.0
	for _, v := range values {
		scale = max(scale, math.Abs(float64(v)))
	}
	worst := 0.0
	for k, id := range indices {
		d := math.Abs(float64(logits[id]) - float64(values[k]))
		worst = max(worst, d/scale)
		if d > relTol*scale {
			t.Fatalf("position %d: logit of token %d (top %d) is %g, PyTorch %g", p, id, k, logits[id], values[k])
		}
	}
	if best := argmax(logits); best != indices[0] && float64(values[0]-values[1]) > 2*relTol*scale {
		t.Fatalf("position %d: argmax %d, PyTorch %d (margin %g)", p, best, indices[0], values[0]-values[1])
	}
	return worst
}

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// compare checks got against the named reference tensor and returns the
// difference relative to the tensor's range.
func compare(t *testing.T, want *safetensors.File, name string, got []float32) float64 {
	t.Helper()
	ref, err := want.Tensor(name)
	if err != nil {
		t.Fatal(err)
	}
	return compareData(t, name, got, ref.Data)
}

func compareData(t *testing.T, name string, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	var diff, scale float64
	for i := range want {
		diff = max(diff, math.Abs(float64(got[i])-float64(want[i])))
		scale = max(scale, math.Abs(float64(want[i])))
	}
	rel := diff / scale
	if rel > relTol {
		t.Fatalf("%s: max |diff| %.3g is %.2g of its range, want <= %g", name, diff, rel, relTol)
	}
	return rel
}
