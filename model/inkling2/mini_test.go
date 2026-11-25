package inkling2

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tokenizer/bpe"
	"github.com/mhpenta/infer/tokenizer/chat"
)

// The tests below need Inkling 2 Mini's weights (19.5 GB, not in CI:
// scripts/fetch_models.sh inkling2-mini) and are skipped with -short.

// miniTol is the largest difference from PyTorch allowed in a hidden state or
// the logits of Inkling 2 Mini, relative to that tensor's largest magnitude.
// It is provisional until goldens from the real checkpoint have been
// compared: the tiny checkpoint's float32 rounding reaches 5e-6 over 10 layer
// visits, and the Mini has 48.
const miniTol = 1e-4

var mini struct {
	model *Model
	err   error
	done  bool
}

// loadMini loads Inkling 2 Mini once for all tests, or skips.
func loadMini(t *testing.T) (*Model, string) {
	t.Helper()
	if testing.Short() {
		t.Skip("Inkling 2 Mini skipped with -short")
	}
	dir := golden.ModelDir(t, "INFER_INKLING2_DIR", "inkling2-mini")
	if _, err := os.Stat(filepath.Join(dir, "model-00001-of-00001.safetensors")); err != nil {
		t.Skipf("Inkling 2 Mini's weights are not in %s: scripts/fetch_models.sh inkling2-mini", dir)
	}
	if !mini.done {
		mini.model, mini.err = Load(dir)
		mini.done = true
	}
	if mini.err != nil {
		t.Fatal(mini.err)
	}
	return mini.model, dir
}

func TestMiniConfig(t *testing.T) {
	m, _ := loadMini(t)
	c := m.Config
	if c.Visits() != 48 || len(m.layers) != 24 || c.EOS != 200006 {
		t.Errorf("%d visits of %d layers, EOS %d", c.Visits(), len(m.layers), c.EOS)
	}
	formats := map[string]int{}
	for _, l := range m.layers {
		formats[fmt.Sprint(l.gateUp[0].Format())]++
	}
	if formats["NVFP4"] != 20 || formats["MXFP8"] != 4 {
		t.Errorf("expert formats %v, want 20 NVFP4 and 4 MXFP8 layers", formats)
	}
}

// TestMiniAnswers renders a chat prompt with no reasoning, decodes greedily,
// and checks the answer.
func TestMiniAnswers(t *testing.T) {
	m, dir := loadMini(t)
	tok, err := bpe.LoadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	tmpl, err := chat.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tmpl.Render([]chat.Message{{Role: "user", Content: "What is 17 * 23? Answer with just the number."}},
		chat.Options{ReasoningEffort: "none"})
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Generate(tok.EncodeRaw(prompt), 48)
	if err != nil {
		t.Fatal(err)
	}
	reply := tok.Decode(out, false)
	t.Logf("reply: %q", reply)
	if content, _, err := tmpl.ParseToolCalls(reply); err != nil || !strings.Contains(content, "391") {
		t.Errorf("reply %q has no 391 (content %q, err %v)", reply, content, err)
	}
}

type miniGolden struct {
	Cases []struct {
		InputIDs  []int `json:"input_ids"`
		Generated []int `json:"generated"`
	} `json:"cases"`
}

// TestMiniMatchesPython compares every layer visit's input, the logits and
// greedy generation with the goldens scripts/dump_golden_inkling2.py writes
// for the real checkpoint, when they are present.
func TestMiniMatchesPython(t *testing.T) {
	m, _ := loadMini(t)
	path := golden.Path(t, "inkling2_mini_model.safetensors")
	if _, err := os.Stat(path); err != nil {
		t.Skip("no goldens for Inkling 2 Mini: scripts/dump_golden_inkling2.py model (needs about 110 GB of memory)")
	}
	want, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	var gold miniGolden
	golden.JSON(t, "inkling2_mini_model.json", &gold)
	for i, c := range gold.Cases {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			p := fmt.Sprintf("case%d.", i)
			states, err := m.HiddenStates(c.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			for j, got := range states {
				if d := relDiff(got.Data, wantTensor(t, want, fmt.Sprintf("%shidden.%d", p, j)).Data); d > miniTol || math.IsNaN(d) {
					t.Errorf("hidden state %d differs by %.3g of its range", j, d)
				}
			}
			logits, err := m.Logits(c.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			if d := relDiff(logits.Data, wantTensor(t, want, p+"logits").Data); d > miniTol || math.IsNaN(d) {
				t.Errorf("logits differ by %.3g of their range", d)
			}
			got, err := m.Generate(c.InputIDs, len(c.Generated))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, c.Generated) {
				t.Errorf("generated %v, want %v", got, c.Generated)
			}
		})
	}
}
