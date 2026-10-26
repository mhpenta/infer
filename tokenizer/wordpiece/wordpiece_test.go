package wordpiece

import (
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
