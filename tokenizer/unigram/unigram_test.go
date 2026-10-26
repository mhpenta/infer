package unigram

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
)

type tokenCase struct {
	Text    string `json:"text"`
	IDs     []int  `json:"ids"`
	Decoded string `json:"decoded"`
}

func checkCases(t *testing.T, tok *Tokenizer, file string) {
	t.Helper()
	var cases []tokenCase
	golden.JSON(t, file, &cases)
	for _, c := range cases {
		name := c.Text
		if len(name) > 40 {
			name = name[:40]
		}
		t.Run(name, func(t *testing.T) {
			if got := tok.Encode(c.Text); !slices.Equal(got, c.IDs) {
				t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
			}
			if got := tok.Decode(c.IDs, true); got != c.Decoded {
				t.Errorf("Decode(%v)\n got %q\nwant %q", c.IDs, got, c.Decoded)
			}
		})
	}
}

func TestT5SmallMatchesPython(t *testing.T) {
	tok, err := LoadDir(golden.ModelDir(t, "INFER_T5_DIR", "t5-small"))
	if err != nil {
		t.Fatal(err)
	}
	checkCases(t, tok, "t5small_tokens.json")
}

// TestWithoutNormalizerMatchesPython checks the same tokenizer with its
// normalizer removed, as some re-saved T5 tokenizers ship.
func TestWithoutNormalizerMatchesPython(t *testing.T) {
	dir := golden.ModelDir(t, "INFER_T5_DIR", "t5-small")
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var tj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tj); err != nil {
		t.Fatal(err)
	}
	tj["normalizer"] = json.RawMessage("null")
	stripped, err := json.Marshal(tj)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, stripped, 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := Load(path, Config{MaxLength: 512, CleanUpSpaces: true})
	if err != nil {
		t.Fatal(err)
	}
	checkCases(t, tok, "t5small_tokens_nonorm.json")
}

func TestModelCasesTokenizeLikePython(t *testing.T) {
	tok, err := LoadDir(golden.ModelDir(t, "INFER_T5_DIR", "t5-small"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text      string `json:"text"`
		InputIDs  []int  `json:"input_ids"`
		OutputIDs []int  `json:"output_ids"`
		Output    string `json:"output"`
	}
	golden.JSON(t, "t5small_model.json", &cases)
	for i, c := range cases {
		if got := tok.Encode(c.Text); !slices.Equal(got, c.InputIDs) {
			t.Errorf("case %d: Encode differs\n got %v\nwant %v", i, got, c.InputIDs)
		}
		if got := tok.Decode(c.OutputIDs, true); got != c.Output {
			t.Errorf("case %d: Decode\n got %q\nwant %q", i, got, c.Output)
		}
	}
}
