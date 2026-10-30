package t5

import (
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/safetensors"
)

// relTol is the largest difference from PyTorch allowed in a hidden state,
// relative to that tensor's largest value. Logits are checked the same way.
const relTol = 2e-5

type modelCase struct {
	Text      string `json:"text"`
	InputIDs  []int  `json:"input_ids"`
	OutputIDs []int  `json:"output_ids"`
	Output    string `json:"output"`
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

// eachDevice runs check on t5-small loaded on each device, with its golden
// cases and reference tensors.
func eachDevice(t *testing.T, check func(t *testing.T, model *Model, cases []modelCase, want *safetensors.File)) {
	for _, device := range testDevices(t) {
		t.Run(device, func(t *testing.T) {
			model, cases, want := loadT5(t, device)
			check(t, model, cases, want)
		})
	}
}

func loadT5(t *testing.T, device string) (*Model, []modelCase, *safetensors.File) {
	t.Helper()
	model, err := LoadWith(golden.ModelDir(t, "INFER_T5_DIR", "t5-small"), Options{Device: device})
	if err != nil {
		t.Fatal(err)
	}
	if device != accel.Auto && !strings.HasPrefix(model.Device(), device) {
		t.Fatalf("loaded on %s, want %s", model.Device(), device)
	}
	t.Cleanup(func() { model.Close() })
	var cases []modelCase
	golden.JSON(t, "t5small_model.json", &cases)
	want, err := safetensors.Open(golden.Path("t5small_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { want.Close() })
	return model, cases, want
}

func TestRelativeBucketMatchesPython(t *testing.T) {
	var table struct {
		NumBuckets     int   `json:"num_buckets"`
		MaxDistance    int   `json:"max_distance"`
		FirstPosition  int   `json:"first_position"`
		Bidirectional  []int `json:"bidirectional"`
		Unidirectional []int `json:"unidirectional"`
	}
	golden.JSON(t, "t5small_buckets.json", &table)
	for i := range table.Bidirectional {
		rel := table.FirstPosition + i
		if got := relativeBucket(rel, true, table.NumBuckets, table.MaxDistance); got != table.Bidirectional[i] {
			t.Errorf("bidirectional bucket(%d) = %d, want %d", rel, got, table.Bidirectional[i])
		}
		if got := relativeBucket(rel, false, table.NumBuckets, table.MaxDistance); got != table.Unidirectional[i] {
			t.Errorf("unidirectional bucket(%d) = %d, want %d", rel, got, table.Unidirectional[i])
		}
	}
}

func TestT5SmallMatchesPython(t *testing.T) {
	eachDevice(t, func(t *testing.T, model *Model, cases []modelCase, want *safetensors.File) {
		for i, c := range cases {
			t.Run(fmt.Sprintf("case%d_%d_in_%d_out", i, len(c.InputIDs), len(c.OutputIDs)-1), func(t *testing.T) {
				enc, err := model.EncoderHiddenStates(c.InputIDs)
				if err != nil {
					t.Fatal(err)
				}
				worst := 0.0
				for j, got := range enc {
					worst = max(worst, compare(t, want, fmt.Sprintf("case%d.enc.hidden.%d", i, j), got.Data))
				}
				// Teacher forcing: the generated sequence minus its last token.
				dec, logits, err := model.DecoderHiddenStates(enc[len(enc)-1], c.OutputIDs[:len(c.OutputIDs)-1])
				if err != nil {
					t.Fatal(err)
				}
				for j, got := range dec {
					worst = max(worst, compare(t, want, fmt.Sprintf("case%d.dec.hidden.%d", i, j), got.Data))
				}
				ref, err := want.Tensor(fmt.Sprintf("case%d.dec.logits", i))
				if err != nil {
					t.Fatal(err)
				}
				worst = max(worst, compareData(t, "logits", logits.Data[:ref.Size()], ref.Data))
				t.Logf("largest difference from PyTorch: %.2g of a tensor's range", worst)
				for p := range logits.Rows() {
					row := logits.Row(p)
					if got := argmax(row); got != c.OutputIDs[p+1] {
						t.Fatalf("position %d: argmax %d, want %d", p, got, c.OutputIDs[p+1])
					}
				}
			})
		}
	})
}

func TestGenerateUncachedMatchesPython(t *testing.T) {
	eachDevice(t, func(t *testing.T, model *Model, cases []modelCase, want *safetensors.File) {
		for i, c := range cases {
			got, err := model.generateUncached(c.InputIDs, 64)
			if err != nil {
				t.Fatal(err)
			}
			if want := c.OutputIDs[1:]; !slices.Equal(got, want) {
				t.Errorf("case %d: generated %v\nwant %v", i, got, want)
			}
		}
	})
}

// compare checks got against the named reference tensor and returns the
// relative difference.
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

func argmax(x []float32) int {
	best := 0
	for i, v := range x {
		if v > x[best] {
			best = i
		}
	}
	return best
}

// TestGenerateMatchesPython checks cached greedy decoding token for token
// against PyTorch (which also uses a cache), including a case that runs to
// the token limit without reaching </s>.
func TestGenerateMatchesPython(t *testing.T) {
	eachDevice(t, func(t *testing.T, model *Model, cases []modelCase, want *safetensors.File) {
		for i, c := range cases {
			got, err := model.Generate(c.InputIDs, 64)
			if err != nil {
				t.Fatal(err)
			}
			if want := c.OutputIDs[1:]; !slices.Equal(got, want) {
				t.Errorf("case %d: generated %v\nwant %v", i, got, want)
			}
		}
	})
}

// TestCachedStepsMatchTeacherForcing compares the cached decoder's logits,
// step by step, with the reference logits from one teacher-forced pass.
func TestCachedStepsMatchTeacherForcing(t *testing.T) {
	eachDevice(t, func(t *testing.T, model *Model, cases []modelCase, want *safetensors.File) {
		for i, c := range cases {
			enc, err := model.Encode(c.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			ref, err := want.Tensor(fmt.Sprintf("case%d.dec.logits", i))
			if err != nil {
				t.Fatal(err)
			}
			st, err := model.newDecoderState(enc, 1) // starts small, so the cache must grow
			if err != nil {
				t.Fatal(err)
			}
			vocab := ref.Cols()
			for p := range ref.Rows() {
				got, err := st.step(c.OutputIDs[p])
				if err != nil {
					t.Fatal(err)
				}
				compareData(t, fmt.Sprintf("case %d step %d", i, p), got, ref.Data[p*vocab:(p+1)*vocab])
			}
			st.release()
		}
	})
}

func TestGenerateRejectsBadInput(t *testing.T) {
	model, _, _ := loadT5(t, accel.Auto)
	for _, ids := range [][]int{nil, {-1, 1}, {model.Config.VocabSize}} {
		if _, err := model.Generate(ids, 4); err == nil {
			t.Errorf("Generate(%v) succeeded, want an error", ids)
		}
	}
}

func BenchmarkGenerate(b *testing.B) {
	for _, device := range testDevices(b) {
		b.Run(device, func(b *testing.B) { benchGenerate(b, device) })
	}
}

func benchGenerate(b *testing.B, device string) {
	model, err := LoadWith(golden.ModelDir(b, "INFER_T5_DIR", "t5-small"), Options{Device: device})
	if err != nil {
		b.Fatal(err)
	}
	defer model.Close()
	var cases []modelCase
	golden.JSON(b, "t5small_model.json", &cases)
	for i, c := range cases {
		b.Run(fmt.Sprintf("case%d_%d_in_%d_out", i, len(c.InputIDs), len(c.OutputIDs)-1), func(b *testing.B) {
			for b.Loop() {
				if _, err := model.Generate(c.InputIDs, 64); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
