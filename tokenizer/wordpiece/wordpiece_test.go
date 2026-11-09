package wordpiece

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
)

func TestEncodeMatchesPython(t *testing.T) {
	tok, err := LoadDir(golden.ModelDir(t, "INFER_MINILM_DIR", "all-MiniLM-L6-v2"))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases(t, "minilm_tokens.json") {
		name := c.Text
		if len(name) > 40 {
			name = name[:40]
		}
		t.Run(name, func(t *testing.T) {
			got := tok.Encode(c.Text)
			if !slices.Equal(got, c.IDs) {
				t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
			}
		})
	}
}

// TestLoadDirMaxLength reads model_max_length as transformers writes it: a
// length, the huge number that stands for none, or nothing.
func TestLoadDirMaxLength(t *testing.T) {
	const vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\n"
	text := "a a a a a a" // 6 words
	for _, c := range []struct {
		config string
		want   int // ids, with [CLS] and [SEP]; 0 for an error
	}{
		{`{"do_lower_case": true, "model_max_length": 5}`, 5},
		{`{"do_lower_case": true, "model_max_length": 512}`, 8},
		{`{"do_lower_case": true, "model_max_length": 1000000000000000019884624838656}`, 8},
		{`{"do_lower_case": true}`, 8},
		{`{"do_lower_case": true, "model_max_length": 1}`, 0},
		{`{"do_lower_case": true, "model_max_length": -4}`, 0},
	} {
		dir := t.TempDir()
		for name, content := range map[string]string{"vocab.txt": vocab, "tokenizer_config.json": c.config} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		tok, err := LoadDir(dir)
		if c.want == 0 {
			if err == nil {
				t.Errorf("%s: LoadDir succeeded", c.config)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.config, err)
			continue
		}
		if got := tok.Encode(text); len(got) != c.want || got[0] != 2 || got[len(got)-1] != 3 {
			t.Errorf("%s: Encode gave %v, want %d ids from [CLS] to [SEP]", c.config, got, c.want)
		}
	}
}
