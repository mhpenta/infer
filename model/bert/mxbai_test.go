package bert

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/pool"
	"github.com/mhpenta/infer/quant"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tokenizer/wordpiece"
)

// mxbaiDims is the Matryoshka truncation applied before binary
// quantization.
const mxbaiDims = 512

// Tolerances for mxbai, a 24-layer model with fp16 weights. Hidden states are
// compared relative to each tensor's largest value; the output that matters
// is the binary code, where a value near zero may land on either side.
const (
	mxbaiRelTol      = 2e-5
	mxbaiMinCosine   = 0.99999
	mxbaiMaxBitFlips = 2 // of 512, per text
)

// mxbaiBinary runs the binary-embedding pipeline: CLS pooling, L2 normalize,
// first mxbaiDims dimensions, binary quantization.
func mxbaiBinary(t *testing.T, m *Model, tok *wordpiece.Tokenizer, text string) ([]float32, []int8) {
	t.Helper()
	hidden, err := m.Forward(tok.Encode(text))
	if err != nil {
		t.Fatal(err)
	}
	v := pool.CLS(hidden)
	pool.L2Normalize(v)
	v = v[:mxbaiDims]
	return v, quant.Binary(v)
}

func loadMxbai(t *testing.T) (*Model, *wordpiece.Tokenizer) {
	t.Helper()
	dir := golden.ModelDir(t, "INFER_MXBAI_DIR", "mxbai-embed-large-v1")
	model, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { model.Close() })
	tok, err := wordpiece.LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	return model, tok
}

func TestMxbaiMatchesPython(t *testing.T) {
	if testing.Short() {
		t.Skip("mxbai is large; skipped with -short")
	}
	model, tok := loadMxbai(t)
	want, err := safetensors.Open(golden.Path("mxbai_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()

	for _, c := range golden.Cases(t, "mxbai_tokens.json") {
		if ids := tok.Encode(c.Text); !slices.Equal(ids, c.IDs) {
			t.Fatalf("tokenizer mismatch for %q", c.Text)
		}
	}

	var cases []struct {
		Text   string `json:"text"`
		IDs    []int  `json:"ids"`
		Binary []int8 `json:"binary"`
	}
	golden.JSON(t, "mxbai_model.json", &cases)
	for i, c := range cases {
		t.Run(fmt.Sprintf("case%d_%d_tokens", i, len(c.IDs)), func(t *testing.T) {
			states, err := model.HiddenStates(c.IDs)
			if err != nil {
				t.Fatal(err)
			}
			for _, j := range []int{0, 1, 12, 24} {
				ref, err := want.Tensor(fmt.Sprintf("case%d.hidden.%d", i, j))
				if err != nil {
					t.Fatal(err)
				}
				rel := relDiff(states[j].Data, ref.Data)
				t.Logf("hidden state %2d: max |diff| / max |ref| = %.2g", j, rel)
				if rel > mxbaiRelTol {
					t.Errorf("hidden state %d differs by %.2g of its range, want <= %g", j, rel, mxbaiRelTol)
				}
			}

			mrl, bits := mxbaiBinary(t, model, tok, c.Text)
			ref, err := want.Tensor(fmt.Sprintf("case%d.mrl", i))
			if err != nil {
				t.Fatal(err)
			}
			if cos := cosine(mrl, ref.Data); cos < mxbaiMinCosine {
				t.Errorf("normalized 512-dim cosine %.7f, want >= %v", cos, mxbaiMinCosine)
			}
			flips := quant.HammingBits(bits, c.Binary)
			t.Logf("binary: %d of %d bits differ from Python", flips, mxbaiDims)
			if flips > mxbaiMaxBitFlips {
				t.Errorf("binary differs in %d bits, want <= %d", flips, mxbaiMaxBitFlips)
			}
		})
	}
}

func TestMxbaiBinaryMatchesPython(t *testing.T) {
	if testing.Short() {
		t.Skip("mxbai is large; skipped with -short")
	}
	model, tok := loadMxbai(t)
	var cases []struct {
		Text   string `json:"text"`
		Binary []int8 `json:"binary"`
	}
	golden.JSON(t, "mxbai_binary.json", &cases)
	total, exact := 0, 0
	for _, c := range cases {
		_, bits := mxbaiBinary(t, model, tok, c.Text)
		flips := quant.HammingBits(bits, c.Binary)
		total += flips
		if flips == 0 {
			exact++
		}
		if flips > mxbaiMaxBitFlips {
			t.Errorf("%q: %d of %d bits differ, want <= %d", c.Text, flips, mxbaiDims, mxbaiMaxBitFlips)
		}
	}
	t.Logf("%d of %d texts match Python bit for bit; %d bits differ in all", exact, len(cases), total)
}

// relDiff returns max |got - want| divided by max |want|.
func relDiff(got, want []float32) float64 {
	var diff, scale float64
	for i := range want {
		diff = max(diff, math.Abs(float64(got[i])-float64(want[i])))
		scale = max(scale, math.Abs(float64(want[i])))
	}
	return diff / scale
}

func BenchmarkMxbaiForward(b *testing.B) {
	dir := golden.ModelDir(b, "INFER_MXBAI_DIR", "mxbai-embed-large-v1")
	model, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	for _, n := range []int{4, 29, 65} {
		ids := make([]int, n)
		for i := range ids {
			ids[i] = 1996
		}
		ids[0], ids[n-1] = 101, 102
		b.Run(fmt.Sprintf("%d_tokens", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := model.Forward(ids); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestMxbaiHalfPanelsIdentical loads mxbai with fp16 and with float32 panels
// and requires identical hidden states: fp16 panels only change what the
// kernels read, not the values they compute with. It needs fp16 SIMD
// kernels (amd64 with AVX2).
func TestMxbaiHalfPanelsIdentical(t *testing.T) {
	if testing.Short() {
		t.Skip("mxbai is large; skipped with -short")
	}
	if !ops.HalfPanels() {
		t.Skip("no fp16-panel SIMD kernels in this build")
	}
	halfModel, tok := loadMxbai(t)
	saved := useHalfPanels
	useHalfPanels = func() bool { return false }
	floatModel, _ := loadMxbai(t)
	useHalfPanels = saved

	// From a short query to the 512-token golden case, so both the 6×16 and
	// the wide tile paths run.
	texts := []string{"apple dividend", "Revenue grew 12.5% year-over-year to $1,234.56 million in Q3 FY2024."}
	var cases []golden.Case
	golden.JSON(t, "mxbai_model.json", &cases)
	texts = append(texts, cases[len(cases)-1].Text)
	for _, text := range texts {
		ids := tok.Encode(text)
		got, err := halfModel.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		want, err := floatModel.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got.Data, want.Data) {
			t.Errorf("%d tokens: fp16 panels change the output (max |diff| %g)", len(ids), maxDiff(got.Data, want.Data))
		}
	}
}
