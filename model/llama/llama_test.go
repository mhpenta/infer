package llama

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/metal"
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

// goldenModel is a checkpoint with PyTorch goldens: its directory under
// testdata/models (or envVar), the goldens' file prefix, and the tolerance
// for the logits at position 0, when it differs from relTol.
type goldenModel struct {
	envVar, dir, prefix string
	firstTol            float64
	config              func(t *testing.T, m *Model) // checks specific to the model
	device              string                       // accel.CPU when empty
}

// on returns g loaded on device.
func (g goldenModel) on(device string) goldenModel {
	g.device = device
	return g
}

func (g goldenModel) key() string { return g.dir + "/" + g.onDevice() }

func (g goldenModel) onDevice() string {
	if g.device == "" {
		return accel.CPU
	}
	return g.device
}

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

// forwardAll runs seq through a fresh state and returns the final hidden
// states. It panics if the GPU fails.
func forwardAll(m *Model, seq []int) *tensor.Tensor {
	st := m.newState(len(seq))
	defer st.release()
	y, err := st.forward(seq, nil)
	if err != nil {
		panic(err)
	}
	return y
}

var (
	miniCPM = goldenModel{"INFER_MINICPM5_DIR", "minicpm5-1b", "minicpm5_1b", 0, checkMiniCPMConfig, ""}

	// Qwen3's first token is an attention sink: by the middle layers its
	// residual values reach 12,000-15,000, against about 100 at every other
	// position, and float32 rounding at that scale leaves its logits 2.2e-5
	// of their range from PyTorch's (after <|im_start|>; 6.1e-6 after the
	// raw-text case's first token). Every later position stays within
	// 1.2e-5.
	qwen3 = goldenModel{"INFER_QWEN3_DIR", "qwen3-1.7b", "qwen3_1.7b", 3e-5, checkQwen3Config, ""}

	goldenModels = []goldenModel{miniCPM, qwen3}
)

// loaded is a model with its golden cases and reference tensors.
type loaded struct {
	model *Model
	cases []modelCase
	want  *safetensors.File
}

var shared struct {
	models map[string]*loaded
}

// load loads a model and its goldens once for all tests, checking its end
// tokens against generate's.
func (g goldenModel) load(t testing.TB) (*Model, []modelCase, *safetensors.File) {
	t.Helper()
	l := shared.models[g.key()]
	if l == nil {
		model, err := LoadWith(golden.ModelDir(t, g.envVar, g.dir), Options{Device: g.onDevice()})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(model.Device(), g.onDevice()) {
			t.Fatalf("loaded on %s, want %s", model.Device(), g.onDevice())
		}
		var gm struct {
			EOS   []int       `json:"eos_token_ids"`
			Cases []modelCase `json:"cases"`
		}
		golden.JSON(t, g.prefix+"_model.json", &gm)
		if !slices.Equal(model.Config.EOS, gm.EOS) {
			t.Fatalf("end tokens %v, want %v", model.Config.EOS, gm.EOS)
		}
		want, err := safetensors.Open(golden.Path(g.prefix + "_model.safetensors"))
		if err != nil {
			t.Fatal(err)
		}
		l = &loaded{model, gm.Cases, want}
		if shared.models == nil {
			shared.models = map[string]*loaded{}
		}
		shared.models[g.key()] = l
	}
	return l.model, l.cases, l.want
}

// release drops a loaded model, so the test binary holds one model at a
// time (CI and Docker machines may have 8 GB).
func (g goldenModel) release() {
	if l := shared.models[g.key()]; l != nil {
		l.model.Close()
		l.want.Close()
		delete(shared.models, g.key())
		runtime.GC()
	}
}

// loadMiniCPM loads MiniCPM5-1B and its goldens.
func loadMiniCPM(t testing.TB) (*Model, []modelCase, *safetensors.File) {
	return miniCPM.load(t)
}

// sequence is a case's prompt and generated tokens but the last: the input
// of the teacher-forced pass the goldens come from.
func (c modelCase) sequence() []int {
	return append(slices.Clone(c.InputIDs), c.OutputIDs[:len(c.OutputIDs)-1]...)
}

// TestMatchesPython runs every comparison with PyTorch for one model on one
// device at a time: its configuration, hidden states, top-64 logits and
// generation.
func TestMatchesPython(t *testing.T) {
	for _, g := range goldenModels {
		for _, device := range testDevices(t) {
			g := g.on(device)
			t.Run(g.key(), func(t *testing.T) {
				defer g.release()
				t.Run("Config", func(t *testing.T) {
					m, _, _ := g.load(t)
					g.config(t, m)
				})
				t.Run("HiddenStates", func(t *testing.T) { checkHiddenStates(t, g) })
				t.Run("TopLogits", func(t *testing.T) { checkTopLogits(t, g) })
				t.Run("Generate", func(t *testing.T) { checkGenerate(t, g) })
				t.Run("Session", func(t *testing.T) { checkSession(t, g) })
			})
		}
	}
}

func checkMiniCPMConfig(t *testing.T, m *Model) {
	c := m.Config
	if c.HeadDim != 128 || c.NumKeyValueHeads != 2 || c.NumAttentionHeads != 16 || c.TieWordEmbeddings || c.qkNorm() {
		t.Errorf("config %+v", c)
	}
	if g := m.gpu; g != nil {
		if g.layers[0].qNorm.Buf != nil || g.lmHead.W == g.embed {
			t.Error("the GPU's weights have query norms or a tied output layer")
		}
		return
	}
	if !m.lmHead.BF16() || !m.layers[0].qkv.BF16() || m.embed.bf16 == nil {
		t.Error("bf16 weights are not kept in bf16")
	}
	if m.layers[0].qNorm != nil {
		t.Error("a Llama model has query norms")
	}
}

func checkQwen3Config(t *testing.T, m *Model) {
	c := m.Config
	if c.HeadDim != 128 || c.NumKeyValueHeads != 8 || c.NumAttentionHeads != 16 || !c.TieWordEmbeddings || !c.qkNorm() {
		t.Errorf("config %+v", c)
	}
	// config.json lists only <|im_end|>; generation_config.json adds <|endoftext|>.
	if !slices.Equal(c.EOS, []int{151645, 151643}) {
		t.Errorf("end tokens %v, want generation_config.json's [151645 151643]", c.EOS)
	}
	if g := m.gpu; g != nil {
		if g.layers[0].qNorm.Buf == nil || g.layers[27].kNorm.Buf == nil || g.lmHead.W != g.embed {
			t.Error("the GPU's weights lack q/k norms or an output layer tied to the embeddings")
		}
		return
	}
	if !m.lmHead.BF16() || m.embed.bf16 == nil || m.layers[0].qNorm == nil || m.layers[27].kNorm == nil {
		t.Error("bf16 weights or q/k norms missing")
	}
}

// checkHiddenStates compares every layer's input and the final
// output with transformers' hidden states, and the full logits of the
// positions where generation starts.
func checkHiddenStates(t *testing.T, g goldenModel) {
	m, cases, want := g.load(t)
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

// checkTopLogits checks the logits at PyTorch's 64 highest-scoring
// tokens, and the argmax, at every position of every case (for the long
// case, every 64th prompt position and all generated ones).
func checkTopLogits(t *testing.T, g goldenModel) {
	m, cases, want := g.load(t)
	for i, c := range cases {
		t.Run(fmt.Sprintf("case%d_%d_in", i, len(c.InputIDs)), func(t *testing.T) {
			if i == longCase && testing.Short() {
				t.Skip("long prompt skipped with -short")
			}
			seq := c.sequence()
			y := forwardAll(m, seq)
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
				tol := relTol
				if p == 0 && g.firstTol > 0 {
					tol = g.firstTol
				}
				m.LogitsInto(logits, tensor.FromData(y.Row(p), 1, y.Cols()))
				worst = max(worst, checkTopK(t, p, logits.Data, values[p], indices[p], tol))
			}
			t.Logf("%d positions; largest difference: %.2g of a row's range", len(positions), worst)
		})
	}
}

// checkGenerate decodes greedily with the KV cache and checks
// every token against PyTorch's generate, and every step's logits against
// the teacher-forced ones at the same position.
func checkGenerate(t *testing.T, g goldenModel) {
	m, cases, want := g.load(t)
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
				worst = max(worst, checkTopK(t, p, logits, values[p], indices[p], relTol))
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

// checkSession runs every case through one Session, three calls each:
//   - the first half of the prompt, leaving the cache holding tokens the
//     next call doesn't share;
//   - the whole prompt, which must reuse that half;
//   - the prompt and the first half of the reply, as the next round of a
//     conversation repeats the last, which must reuse everything but its
//     last token.
//
// The full and continued prompts must generate PyTorch's tokens, with
// every step's logits within relTol of the teacher-forced ones and equal to
// GenerateFunc's from an empty cache: bit-identical on the CPU, within
// sessionTol on the GPU, which computes prompts and decoding steps with
// different kernels.
func checkSession(t *testing.T, g goldenModel) {
	m, cases, want := g.load(t)
	s := m.NewSession()
	for i, c := range cases {
		if i == longCase && testing.Short() {
			continue
		}
		seq := c.sequence()
		values, indices := topK(t, want, i, len(seq))
		// generate continues seq[:n] and checks every token against the
		// goldens.
		generate := func(n int) {
			t.Helper()
			var fresh [][]float32
			if err := m.GenerateFunc(seq[:n], len(seq)+1-n, func(_ int, logits []float32) bool {
				fresh = append(fresh, slices.Clone(logits))
				return true
			}); err != nil {
				t.Fatal(err)
			}
			var got []int
			worst, fromFresh := 0.0, 0.0
			err := s.GenerateWith(seq[:n], len(seq)+1-n, nil, func(token int, logits []float32) bool {
				p := n - 1 + len(got)
				worst = max(worst, checkTopK(t, p, logits, values[p], indices[p], relTol))
				if len(got) < len(fresh) {
					d, same := sameLogits(m, logits, fresh[len(got)])
					if !same {
						t.Fatalf("case %d after %d tokens, step %d: logits differ from an empty cache's by %.2g of the row's range", i, n, len(got), d)
					}
					fromFresh = max(fromFresh, d)
				}
				got = append(got, token)
				return true
			})
			if err != nil {
				t.Fatal(err)
			}
			if want := c.OutputIDs[n-len(c.InputIDs):]; !slices.Equal(got, want) {
				t.Fatalf("case %d after %d tokens: generated %v\nwant %v", i, n, got, want)
			}
			t.Logf("case %d after %d tokens, %d reused: largest step difference %.2g, from an empty cache's %.2g", i, n, s.Reused(), worst, fromFresh)
		}

		half := len(c.InputIDs) / 2
		if err := s.GenerateWith(c.InputIDs[:half], 4, nil, func(int, []float32) bool { return true }); err != nil {
			t.Fatal(err)
		}
		generate(len(c.InputIDs))
		if s.Reused() < half {
			t.Errorf("case %d: reused %d positions of the whole prompt, want at least %d", i, s.Reused(), half)
		}
		n := len(c.InputIDs) + len(c.OutputIDs)/2
		generate(n)
		if s.Reused() != n-1 {
			t.Errorf("case %d: reused %d positions of the continued prompt, want %d", i, s.Reused(), n-1)
		}
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
// values, within tol of the row's largest golden value, and requires the
// same argmax unless PyTorch's top two are within twice that of each other.
// It returns the largest difference relative to the row's largest value.
func checkTopK(t *testing.T, p int, logits, values []float32, indices []int, tol float64) float64 {
	t.Helper()
	scale := 0.0
	for _, v := range values {
		scale = max(scale, math.Abs(float64(v)))
	}
	worst := 0.0
	for k, id := range indices {
		d := math.Abs(float64(logits[id]) - float64(values[k]))
		worst = max(worst, d/scale)
		if d > tol*scale {
			t.Fatalf("position %d: logit of token %d (top %d) is %g, PyTorch %g", p, id, k, logits[id], values[k])
		}
	}
	if best := argmax(logits); best != indices[0] && float64(values[0]-values[1]) > 2*tol*scale {
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

// sessionTol is how far a Session's logits on the GPU may be from a fresh
// run's, relative to the row's largest value.
const sessionTol = 1e-5

// sameLogits returns the largest difference between rows of logits a and
// b, relative to b's largest value, and whether they count as the same: bit
// for bit on the CPU, within sessionTol on the GPU.
func sameLogits(m *Model, a, b []float32) (float64, bool) {
	if len(a) != len(b) {
		return math.Inf(1), false
	}
	var peak, worst float64
	for i := range b {
		peak = max(peak, math.Abs(float64(b[i])))
		worst = max(worst, math.Abs(float64(a[i])-float64(b[i])))
	}
	worst /= peak
	if m.gpu == nil {
		return worst, bitsEqual(a, b)
	}
	return worst, worst <= sessionTol
}

func bitsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}
