package bpe

import (
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
)

type tokenCase struct {
	Text               string `json:"text"`
	IDs                []int  `json:"ids"`
	IDsWithSpecial     []int  `json:"ids_with_special"`
	Decoded            string `json:"decoded"`
	DecodedSkipSpecial string `json:"decoded_skip_special"`
}

func loadMiniCPM(t testing.TB) *Tokenizer {
	t.Helper()
	tok, err := LoadDir(golden.ModelDir(t, "INFER_MINICPM5_DIR", "minicpm5-1b"))
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func checkCases(t *testing.T, tok *Tokenizer, file string) {
	t.Helper()
	var cases []tokenCase
	golden.JSON(t, file, &cases)
	for i, c := range cases {
		name := fmt.Sprintf("%d_%.30q", i, c.Text)
		t.Run(name, func(t *testing.T) {
			if got := tok.EncodeRaw(c.Text); !slices.Equal(got, c.IDs) {
				t.Errorf("EncodeRaw(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
			}
			if got := tok.Encode(c.Text); !slices.Equal(got, c.IDsWithSpecial) {
				t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDsWithSpecial)
			}
			if got := tok.Decode(c.IDs, false); got != c.Decoded {
				t.Errorf("Decode(%v, false)\n got %q\nwant %q", c.IDs, got, c.Decoded)
			}
			if got := tok.Decode(c.IDs, true); got != c.DecodedSkipSpecial {
				t.Errorf("Decode(%v, true)\n got %q\nwant %q", c.IDs, got, c.DecodedSkipSpecial)
			}
		})
	}
}

func TestMiniCPM5MatchesPython(t *testing.T) {
	checkCases(t, loadMiniCPM(t), "minicpm5_1b_tokens.json")
}

// TestMiniCPM5RandomTextMatchesPython runs 400 seeded random strings mixing
// scripts, Unicode spaces and digits, combining marks, contractions and
// fragments of added tokens (see scripts/dump_golden_llama.py).
func TestMiniCPM5RandomTextMatchesPython(t *testing.T) {
	checkCases(t, loadMiniCPM(t), "minicpm5_1b_tokens_random.json")
}

// TestMiniCPM5ChatTextMatchesPython encodes the text the chat template
// rendered; the template itself is the chat package's to reproduce.
func TestMiniCPM5ChatTextMatchesPython(t *testing.T) {
	tok := loadMiniCPM(t)
	var cases []struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	}
	golden.JSON(t, "minicpm5_1b_chat.json", &cases)
	for i, c := range cases {
		if got := tok.EncodeRaw(c.Text); !slices.Equal(got, c.IDs) {
			t.Errorf("case %d: EncodeRaw(%q)\n got %v\nwant %v", i, c.Text, got, c.IDs)
		}
		if got := tok.Decode(c.IDs, false); got != c.Text {
			t.Errorf("case %d: Decode = %q, want %q", i, got, c.Text)
		}
	}
	for _, special := range []string{"<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>"} {
		if _, ok := tok.TokenID(special); !ok {
			t.Errorf("TokenID(%q) not found", special)
		}
	}
}

func TestSplitWords(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"I'm they'LL don'T", []string{"I", "'m", " they", "'LL", " don", "'T"}},
		{"'ſ 'x", []string{"'ſ", " '", "x"}},           // a word takes one leading non-letter, not two
		{"a  b", []string{"a", " ", " b"}},             // \s+(?!\S) leaves one space to the word
		{"a   ", []string{"a", "   "}},                 // whitespace at the end
		{"x\n\n  y", []string{"x", "\n\n", " ", " y"}}, // through the last newline, then as above
		{" \t\n \n z", []string{" \t\n \n", " z"}},
		{"!!! ???\n\nok", []string{"!!!", " ???\n\n", "ok"}},
		{"$5 (x)", []string{"$", "5", " (", "x", ")"}},
		{" word　", []string{" word", "　"}},
		{"été", []string{"e", "́té"}}, // a combining mark is not a letter
		{"\r\nx", []string{"\r\n", "x"}},
		{"​", []string{"​"}}, // zero-width space is not White_Space
	}
	for _, tt := range tests {
		if got := splitWords(nil, tt.in); !slices.Equal(got, tt.want) {
			t.Errorf("splitWords(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestSplitDigits(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"1234567", []string{"123", "456", "7"}},
		{"a12b", []string{"a", "12", "b"}},
		{"٣٤٥٦ ½²", []string{"٣٤٥", "٦", " ", "½²"}},
		{"none", []string{"none"}},
	}
	for _, tt := range tests {
		if got := splitDigits(3)(nil, tt.in); !slices.Equal(got, tt.want) {
			t.Errorf("splitDigits(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestLossyString checks invalid UTF-8 against Rust's from_utf8_lossy, which
// replaces each maximal subpart of an ill-formed sequence once.
func TestLossyString(t *testing.T) {
	tests := []struct{ in, want string }{
		{"ok", "ok"},
		{"Hello \xF0\x90\x80World", "Hello �World"}, // truncated 4-byte sequence
		{"\xff\xfe", "��"},                          // two invalid bytes
		{"\xe0\x80\x80", "���"},                     // E0 needs A0..BF next
		{"\xed\xa0\x80", "���"},                     // a surrogate
		{"\xf4\x90\x80\x80", "����"},                // above U+10FFFF
		{"\xe2\x82", "�"},                           // truncated at the end
		{"a\xc3", "a�"},                             // ditto
		{"\xc3\xa9\xe2\x82\xac", "é€"},              // valid
		{"\xf0\x9f\x98\x80\xf0\x9f\x98", "😀�"},      // a split emoji
	}
	for _, tt := range tests {
		if got := lossyString([]byte(tt.in)); got != tt.want {
			t.Errorf("lossyString(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestDecodeSplitCharacter(t *testing.T) {
	tok := loadMiniCPM(t)
	ids := tok.EncodeRaw("😀")
	if len(ids) < 2 {
		t.Skipf("😀 is one token (%v); nothing to split", ids)
	}
	if got := tok.Decode(ids[:len(ids)-1], false); !strings.HasSuffix(got, "�") {
		t.Errorf("decoding part of 😀 gave %q, want a replacement character", got)
	}
}

func TestConcurrentEncode(t *testing.T) {
	tok := loadMiniCPM(t)
	text := "Concurrent tokenizers share a cache of pieces: 12345, naïve café, 日本語."
	want := tok.Encode(text)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 50 {
				if got := tok.Encode(text); !slices.Equal(got, want) {
					t.Error("concurrent Encode differs")
					return
				}
			}
		})
	}
	wg.Wait()
}

func BenchmarkEncode(b *testing.B) {
	tok := loadMiniCPM(b)
	text := strings.Repeat("Net sales increased 8% to $94.9 billion, driven by record Services revenue. ", 200)
	b.SetBytes(int64(len(text)))
	for b.Loop() {
		tok.EncodeRaw(text)
	}
}
