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

// TestMetaspaceWithoutPrefix encodes words Metaspace leaves without its
// replacement in front (prepend_scheme "never", or "first" after an added
// token), including ones shorter than the replacement's three bytes.
func TestMetaspaceWithoutPrefix(t *testing.T) {
	load := func(scheme string) *Tokenizer {
		t.Helper()
		src := `{"added_tokens": [{"id": 4, "content": "<x>", "special": true}],
			"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "` + scheme + `", "split": true},
			"decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "` + scheme + `", "split": true},
			"model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0], ["a", -1], ["▁", -2], ["▁a", -1], ["<x>", 0]]}}`
		path := filepath.Join(t.TempDir(), "tokenizer.json")
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		tok, err := Load(path, Config{})
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	for _, c := range []struct {
		scheme, text string
		want         []int
	}{
		{"never", "a", []int{1}},
		{"never", "a a", []int{1, 3}},
		{"never", " a", []int{3}},
		{"never", "a ", []int{1, 2}},
		{"first", "a", []int{3}},
		{"first", "<x>a", []int{4, 1}},
		{"first", "<x>a a", []int{4, 1, 3}},
		{"always", "a", []int{3}},
		{"always", "<x>a", []int{4, 3}},
		{"always", "  a", []int{2, 3}}, // "▁▁a" already starts with it
	} {
		if got := load(c.scheme).Encode(c.text); !slices.Equal(got, c.want) {
			t.Errorf("%s: Encode(%q) = %v, want %v", c.scheme, c.text, got, c.want)
		}
	}
}
