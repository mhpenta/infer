// Package bpe implements byte-level BPE tokenizers (GPT-2 style, as used by
// MiniCPM and most decoder-only models), loaded from a Hugging Face
// tokenizer.json.
//
// It reproduces the Hugging Face fast tokenizer for single sequences:
//
//  1. Added tokens ("<s>", "<|im_start|>", "<think>", ...) in the input are
//     matched first and map straight to their ids.
//  2. Pre-tokenization: the Split patterns, each cutting every piece from the
//     step before (see splitPatterns), then ByteLevel, which writes each
//     byte of a piece as one character of GPT-2's byte alphabet.
//  3. BPE: each piece starts as one token per byte and merges pairs, lowest
//     rank first, until no merge applies.
//  4. The post-processor template (MiniCPM prepends "<s>"), for Encode.
//
// Decode maps the byte alphabet back to bytes and decodes them as UTF-8,
// replacing invalid sequences as Rust's from_utf8_lossy does. Components
// outside this pipeline are rejected rather than approximated.
package bpe

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/mhpenta/infer/tokenizer/internal/hf"
)

// cacheSize is how many pieces' tokens are remembered, as in the tokenizers
// library; once full, new pieces are not cached.
const cacheSize = 10000

// Config controls decoding.
type Config struct {
	// CleanUpSpaces removes spaces before punctuation and in English
	// contractions when decoding (clean_up_tokenization_spaces).
	CleanUpSpaces bool
}

// Tokenizer turns text into ids and back. It is safe for concurrent use.
type Tokenizer struct {
	config Config

	vocab   map[string]int // tokens as tokenizer.json writes them, added tokens included
	tokens  []string       // by id
	special []bool         // by id
	merges  map[uint64]merge
	byteIDs [256]int32 // the token for each single byte

	added    *hf.Added
	splits   []splitFunc
	template hf.Template

	mu    sync.Mutex
	cache map[string][]int
}

// LoadDir loads tokenizer.json and reads clean_up_tokenization_spaces from
// tokenizer_config.json, if present.
func LoadDir(dir string) (*Tokenizer, error) {
	config := Config{CleanUpSpaces: true} // transformers' default
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	switch {
	case err == nil:
		var tc struct {
			CleanUp *bool `json:"clean_up_tokenization_spaces"`
		}
		if err := json.Unmarshal(raw, &tc); err != nil {
			return nil, fmt.Errorf("bpe: parse tokenizer_config.json: %w", err)
		}
		if tc.CleanUp != nil {
			config.CleanUpSpaces = *tc.CleanUp
		}
	case !errors.Is(err, os.ErrNotExist):
		return nil, fmt.Errorf("bpe: %w", err)
	}
	return Load(filepath.Join(dir, "tokenizer.json"), config)
}

// Load reads a tokenizer.json whose model is byte-level BPE.
func Load(path string, config Config) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("bpe: %w", err)
	}
	var tj struct {
		AddedTokens   []hf.AddedToken `json:"added_tokens"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Decoder       json.RawMessage `json:"decoder"`
		Model         struct {
			Type                    string            `json:"type"`
			Vocab                   map[string]int    `json:"vocab"`
			Merges                  []json.RawMessage `json:"merges"`
			Dropout                 *float64          `json:"dropout"`
			ContinuingSubwordPrefix *string           `json:"continuing_subword_prefix"`
			EndOfWordSuffix         *string           `json:"end_of_word_suffix"`
			ByteFallback            bool              `json:"byte_fallback"`
			IgnoreMerges            bool              `json:"ignore_merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("bpe: parse %s: %w", path, err)
	}
	m := tj.Model
	switch {
	case m.Type != "BPE":
		return nil, fmt.Errorf("bpe: %s has a %q model", path, m.Type)
	case m.Dropout != nil && *m.Dropout != 0:
		return nil, errors.New("bpe: dropout is not supported")
	case m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "",
		m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "":
		return nil, errors.New("bpe: subword prefixes and suffixes are not supported")
	case m.ByteFallback:
		return nil, errors.New("bpe: byte_fallback is not supported")
	case m.IgnoreMerges:
		return nil, errors.New("bpe: ignore_merges is not supported")
	case !hf.IsNull(tj.Normalizer):
		return nil, errors.New("bpe: normalizers are not supported")
	}

	t := &Tokenizer{config: config, vocab: map[string]int{}, merges: map[uint64]merge{}, cache: map[string][]int{}}
	size := 0
	for tok, id := range m.Vocab {
		if id < 0 {
			return nil, fmt.Errorf("bpe: token %q has id %d", tok, id)
		}
		size = max(size, id+1)
	}
	for _, a := range tj.AddedTokens {
		size = max(size, a.ID+1)
	}
	t.tokens = make([]string, size)
	t.special = make([]bool, size)
	for tok, id := range m.Vocab {
		t.vocab[tok], t.tokens[id] = id, tok
	}
	for _, a := range tj.AddedTokens {
		// The added vocabulary takes precedence when mapping ids to tokens.
		t.vocab[a.Content], t.tokens[a.ID], t.special[a.ID] = a.ID, a.Content, a.Special
	}
	for b, r := range byteChars {
		id, ok := m.Vocab[string(r)]
		if !ok {
			return nil, fmt.Errorf("bpe: byte %#x (%q) has no token", b, r)
		}
		t.byteIDs[b] = int32(id)
	}
	for rank, raw := range m.Merges {
		a, b, err := parseMerge(raw)
		if err != nil {
			return nil, fmt.Errorf("bpe: merge %d: %w", rank, err)
		}
		ia, okA := m.Vocab[a]
		ib, okB := m.Vocab[b]
		merged, okM := m.Vocab[a+b]
		if !okA || !okB || !okM {
			return nil, fmt.Errorf("bpe: merge %d (%q, %q) uses a token outside the vocabulary", rank, a, b)
		}
		key := pairKey(int32(ia), int32(ib))
		if _, dup := t.merges[key]; !dup {
			t.merges[key] = merge{rank: int32(rank), id: int32(merged)}
		}
	}

	if t.added, err = hf.NewAdded(tj.AddedTokens); err != nil {
		return nil, fmt.Errorf("bpe: %w", err)
	}
	if err := t.parsePreTokenizer(tj.PreTokenizer); err != nil {
		return nil, err
	}
	if t.template, err = hf.ParseTemplate(tj.PostProcessor); err != nil {
		return nil, fmt.Errorf("bpe: %w", err)
	}
	if err := parseDecoder(tj.Decoder); err != nil {
		return nil, err
	}
	return t, nil
}

// parseMerge reads a merge as a pair ["a", "b"] or, in older files, "a b".
func parseMerge(raw json.RawMessage) (string, string, error) {
	var pair []string
	if err := json.Unmarshal(raw, &pair); err == nil {
		if len(pair) != 2 {
			return "", "", fmt.Errorf("%s is not a pair", raw)
		}
		return pair[0], pair[1], nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", "", fmt.Errorf("%s is neither a pair nor a string", raw)
	}
	a, b, ok := strings.Cut(s, " ")
	if !ok {
		return "", "", fmt.Errorf("%q has no space", s)
	}
	return a, b, nil
}

// parsePreTokenizer accepts Split patterns this package implements, then a
// ByteLevel step that only maps bytes (no prefix space, no regex of its own).
func (t *Tokenizer) parsePreTokenizer(raw json.RawMessage) error {
	var p struct {
		Type          string            `json:"type"`
		PreTokenizers []json.RawMessage `json:"pretokenizers"`
	}
	if hf.IsNull(raw) {
		return errors.New("bpe: a ByteLevel pre_tokenizer is required")
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("bpe: parse pre_tokenizer: %w", err)
	}
	steps := []json.RawMessage{raw}
	if p.Type == "Sequence" {
		steps = p.PreTokenizers
	}
	for i, step := range steps {
		var s struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex  *string `json:"Regex"`
				String *string `json:"String"`
			} `json:"pattern"`
			Behavior       string `json:"behavior"`
			Invert         bool   `json:"invert"`
			AddPrefixSpace bool   `json:"add_prefix_space"`
			UseRegex       bool   `json:"use_regex"`
		}
		if err := json.Unmarshal(step, &s); err != nil {
			return fmt.Errorf("bpe: parse pre_tokenizer: %w", err)
		}
		switch s.Type {
		case "Split":
			if s.Pattern.Regex == nil || s.Behavior != "Isolated" || s.Invert {
				return fmt.Errorf("bpe: only Isolated Split on a regex is supported, got %s", step)
			}
			fn, ok := splitPatterns[*s.Pattern.Regex]
			if !ok {
				return fmt.Errorf("bpe: Split pattern %q is not supported", *s.Pattern.Regex)
			}
			t.splits = append(t.splits, fn)
		case "ByteLevel":
			if i != len(steps)-1 || s.AddPrefixSpace || s.UseRegex {
				return errors.New("bpe: only a final ByteLevel step without add_prefix_space or use_regex is supported")
			}
			return nil
		default:
			return fmt.Errorf("bpe: pre_tokenizer %q is not supported", s.Type)
		}
	}
	return errors.New("bpe: the pre_tokenizer does not end with ByteLevel")
}

func parseDecoder(raw json.RawMessage) error {
	var d struct {
		Type string `json:"type"`
	}
	if hf.IsNull(raw) {
		return errors.New("bpe: a ByteLevel decoder is required")
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("bpe: parse decoder: %w", err)
	}
	if d.Type != "ByteLevel" {
		return fmt.Errorf("bpe: decoder %q is not supported", d.Type)
	}
	return nil
}

// Encode returns the ids for text, including the post-processor's tokens
// (add_special_tokens=True in transformers).
func (t *Tokenizer) Encode(text string) []int {
	return t.template.Apply(t.EncodeRaw(text))
}

// EncodeRaw returns the ids for text without the post-processor's tokens
// (add_special_tokens=False), as for text a chat template already wrapped.
// Added tokens written in the text, such as "<s>", are still recognized.
func (t *Tokenizer) EncodeRaw(text string) []int {
	var ids []int
	var pieces, next []string
	for _, seg := range t.added.Split(text, nil) {
		if seg.Token != nil {
			ids = append(ids, seg.Token.ID)
			continue
		}
		pieces = append(pieces[:0], seg.Text)
		for _, split := range t.splits {
			next = next[:0]
			for _, p := range pieces {
				next = split(next, p)
			}
			pieces, next = next, pieces
		}
		for _, p := range pieces {
			ids = t.encodePiece(ids, p)
		}
	}
	return ids
}

// encodePiece appends the tokens of one pre-tokenized piece.
func (t *Tokenizer) encodePiece(dst []int, piece string) []int {
	if len(piece) == 1 {
		return append(dst, int(t.byteIDs[piece[0]]))
	}
	t.mu.Lock()
	cached, ok := t.cache[piece]
	t.mu.Unlock()
	if ok {
		return append(dst, cached...)
	}
	start := len(dst)
	ids := make([]int32, len(piece))
	for i := range len(piece) {
		ids[i] = t.byteIDs[piece[i]]
	}
	dst = t.mergeWord(dst, ids)
	t.mu.Lock()
	if len(t.cache) < cacheSize {
		t.cache[piece] = append([]int(nil), dst[start:]...)
	}
	t.mu.Unlock()
	return dst
}

// Decode turns ids back into text, skipping special tokens if asked. Ids
// outside the vocabulary are skipped.
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string {
	var buf []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.tokens) || (skipSpecial && t.special[id]) {
			continue
		}
		buf = appendTokenBytes(buf, t.tokens[id])
	}
	out := lossyString(buf)
	if t.config.CleanUpSpaces {
		out = hf.CleanUp(out)
	}
	return out
}

// appendTokenBytes appends the bytes a token stands for: each character
// mapped back through the byte alphabet or, if any character is outside it
// (as in some added tokens), the token's own UTF-8, as the tokenizers
// library's ByteLevel decoder does.
func appendTokenBytes(dst []byte, tok string) []byte {
	start := len(dst)
	for _, r := range tok {
		b, ok := charBytes[r]
		if !ok {
			return append(dst[:start], tok...)
		}
		dst = append(dst, b)
	}
	return dst
}

// lossyString decodes b as UTF-8, replacing each maximal invalid subpart
// with U+FFFD, as Rust's String::from_utf8_lossy does.
func lossyString(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var s strings.Builder
	for len(b) > 0 {
		r, size := utf8.DecodeRune(b)
		if r != utf8.RuneError || size > 1 {
			s.Write(b[:size])
		} else {
			s.WriteRune(utf8.RuneError)
			size = invalidPrefix(b)
		}
		b = b[size:]
	}
	return s.String()
}

// invalidPrefix returns the length of the maximal subpart of an ill-formed
// UTF-8 sequence at the start of b (Unicode §3.9): a lead byte and the
// continuation bytes that could still extend it, or one byte.
func invalidPrefix(b []byte) int {
	lo, hi, need := byte(0x80), byte(0xbf), 0
	switch c := b[0]; {
	case 0xc2 <= c && c <= 0xdf:
		need = 1
	case c == 0xe0:
		lo, need = 0xa0, 2
	case c == 0xed:
		hi, need = 0x9f, 2
	case 0xe1 <= c && c <= 0xef:
		need = 2
	case c == 0xf0:
		lo, need = 0x90, 3
	case c == 0xf4:
		hi, need = 0x8f, 3
	case 0xf1 <= c && c <= 0xf3:
		need = 3
	default:
		return 1
	}
	n := 1
	for ; n <= need && n < len(b); n++ {
		if b[n] < lo || b[n] > hi {
			break
		}
		lo, hi = 0x80, 0xbf // only the second byte has a narrower range
	}
	return n
}

// TokenID returns the id of a vocabulary entry or added token, written as in
// tokenizer.json (byte-level tokens in GPT-2's byte alphabet).
func (t *Tokenizer) TokenID(token string) (int, bool) {
	id, ok := t.vocab[token]
	return id, ok
}

// VocabSize returns the number of ids, added tokens included.
func (t *Tokenizer) VocabSize() int { return len(t.tokens) }
