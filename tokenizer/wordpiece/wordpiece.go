// Package wordpiece implements BERT's tokenizer: the BERT normalizer and
// pre-tokenizer followed by greedy longest-match-first WordPiece.
//
// It reproduces the Hugging Face fast BertTokenizer for single sequences:
//
//  1. Literal special tokens ("[CLS]", "[SEP]", ...) in the input are matched
//     before normalization and map straight to their ids.
//  2. Clean text: drop NUL, U+FFFD and control/format characters; turn
//     whitespace into spaces.
//  3. Surround CJK ideographs with spaces so each becomes its own word.
//  4. If lowercasing: strip accents (NFD, drop non-spacing marks), lowercase.
//  5. Split on whitespace, and isolate every punctuation character.
//  6. WordPiece each word, with "##" marking word continuations; words that
//     cannot be covered, or are over 100 characters, become [UNK].
//  7. Truncate to MaxLength-2 and wrap in [CLS] ... [SEP].
package wordpiece

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// maxWordChars matches BERT's max_input_chars_per_word.
const maxWordChars = 100

// specialTokens are BERT's added tokens, matched literally in the input.
var specialTokens = []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}

// Config controls normalization and length.
type Config struct {
	// Lowercase lowercases the text and strips accents, as BERT does when
	// strip_accents is unset (do_lower_case in tokenizer_config.json).
	Lowercase bool
	// MaxLength is the maximum sequence length including [CLS] and [SEP]
	// (model_max_length in tokenizer_config.json).
	MaxLength int
}

// Tokenizer turns text into BERT input ids.
type Tokenizer struct {
	config  Config
	vocab   map[string]int
	special map[string]int
	unkID   int
	clsID   int
	sepID   int
}

// Load reads a vocab.txt (one token per line, id = line number).
func Load(vocabPath string, config Config) (*Tokenizer, error) {
	if config.MaxLength < 2 {
		return nil, fmt.Errorf("wordpiece: MaxLength %d leaves no room for [CLS] and [SEP]", config.MaxLength)
	}
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("wordpiece: %w", err)
	}
	defer f.Close()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(f)
	for id := 0; scanner.Scan(); id++ {
		token := strings.TrimRight(scanner.Text(), "\r")
		if _, dup := vocab[token]; !dup {
			vocab[token] = id
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("wordpiece: read %s: %w", vocabPath, err)
	}

	t := &Tokenizer{config: config, vocab: vocab, special: make(map[string]int)}
	for _, token := range specialTokens {
		id, ok := vocab[token]
		if !ok {
			return nil, fmt.Errorf("wordpiece: vocab %s has no %s", vocabPath, token)
		}
		t.special[token] = id
	}
	t.unkID, t.clsID, t.sepID = t.special["[UNK]"], t.special["[CLS]"], t.special["[SEP]"]
	return t, nil
}

// LoadDir loads vocab.txt and reads do_lower_case and model_max_length from
// tokenizer_config.json in a model directory.
func LoadDir(dir string) (*Tokenizer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("wordpiece: %w", err)
	}
	var tc struct {
		DoLowerCase    *bool `json:"do_lower_case"`
		ModelMaxLength int   `json:"model_max_length"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, fmt.Errorf("wordpiece: parse tokenizer_config.json: %w", err)
	}
	if tc.DoLowerCase == nil {
		return nil, errors.New("wordpiece: tokenizer_config.json has no do_lower_case")
	}
	return Load(filepath.Join(dir, "vocab.txt"), Config{Lowercase: *tc.DoLowerCase, MaxLength: tc.ModelMaxLength})
}

// Encode returns the input ids for text, including [CLS] and [SEP].
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for text != "" {
		pos, token := t.nextSpecial(text)
		for _, word := range t.words(text[:pos]) {
			ids = t.appendWordPieces(ids, word)
		}
		if token == "" {
			break
		}
		ids = append(ids, t.special[token])
		text = text[pos+len(token):]
	}
	ids = ids[:min(len(ids), t.config.MaxLength-2)]
	return append(append([]int{t.clsID}, ids...), t.sepID)
}

// nextSpecial finds the leftmost special token in text, preferring the
// longest at a position. It returns len(text) and "" when there is none.
func (t *Tokenizer) nextSpecial(text string) (int, string) {
	best, bestToken := len(text), ""
	for _, token := range specialTokens {
		i := strings.Index(text, token)
		if i >= 0 && (i < best || (i == best && len(token) > len(bestToken))) {
			best, bestToken = i, token
		}
	}
	return best, bestToken
}

// words normalizes text and splits it into whitespace- and
// punctuation-separated words.
func (t *Tokenizer) words(text string) []string {
	var b strings.Builder
	for _, r := range text {
		switch {
		case r == 0 || r == utf8.RuneError || isControl(r):
		case isWhitespace(r):
			b.WriteByte(' ')
		case isCJK(r):
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
		default:
			b.WriteRune(r)
		}
	}
	normalized := b.String()
	if t.config.Lowercase {
		normalized = strings.ToLower(stripAccents(normalized))
	}

	var words []string
	start := -1
	for i, r := range normalized {
		switch {
		case isWhitespace(r):
			if start >= 0 {
				words = append(words, normalized[start:i])
				start = -1
			}
		case isPunctuation(r):
			if start >= 0 {
				words = append(words, normalized[start:i])
				start = -1
			}
			words = append(words, normalized[i:i+utf8.RuneLen(r)])
		default:
			if start < 0 {
				start = i
			}
		}
	}
	if start >= 0 {
		words = append(words, normalized[start:])
	}
	return words
}

// appendWordPieces appends the WordPiece ids for one word, or [UNK] if the
// word is too long or cannot be fully covered by the vocabulary.
func (t *Tokenizer) appendWordPieces(ids []int, word string) []int {
	if utf8.RuneCountInString(word) > maxWordChars {
		return append(ids, t.unkID)
	}
	var pieces []int
	for start := 0; start < len(word); {
		end := len(word)
		id := -1
		for end > start {
			piece := word[start:end]
			if start > 0 {
				piece = "##" + piece
			}
			if v, ok := t.vocab[piece]; ok {
				id = v
				break
			}
			_, size := utf8.DecodeLastRuneInString(word[start:end])
			end -= size
		}
		if id < 0 {
			return append(ids, t.unkID)
		}
		pieces = append(pieces, id)
		start = end
	}
	return append(ids, pieces...)
}

// stripAccents decomposes to NFD and drops non-spacing marks.
func stripAccents(s string) string {
	var b strings.Builder
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func isWhitespace(r rune) bool {
	return unicode.IsSpace(r)
}

// isControl matches BERT: \t, \n and \r count as whitespace, not control;
// otherwise any "C" category character is dropped.
func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.Is(unicode.C, r)
}

// isPunctuation matches BERT: all ASCII non-alphanumeric printable
// characters (including symbols like $ and ^) plus Unicode punctuation.
func isPunctuation(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// isCJK reports whether r is in a CJK Unified Ideographs block, as BERT
// defines it (Hangul, Hiragana and Katakana are not included).
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}
