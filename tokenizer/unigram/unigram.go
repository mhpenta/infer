// Package unigram implements the SentencePiece Unigram tokenizer used by T5
// and its relatives, loaded from a Hugging Face tokenizer.json.
//
// It reproduces the Hugging Face fast tokenizer for single sequences:
//
//  1. Added tokens ("<pad>", "</s>", "<extra_id_0>", ...) in the raw input are
//     matched first and map straight to their ids.
//  2. The normalizer, if any: SentencePiece's precompiled character map
//     (NFKC-style folding) and Replace rules.
//  3. Pre-tokenization: split on whitespace, then Metaspace, which marks the
//     start of each word with "▁".
//  4. Unigram segmentation: the Viterbi path through all vocabulary pieces
//     that maximizes the sum of their log-probability scores. A character no
//     piece covers becomes <unk>; consecutive <unk>s merge into one.
//  5. Truncation to MaxLength, then the post-processor template (T5 appends
//     "</s>").
//
// Decode reverses the Metaspace marks and, like transformers, can skip
// special tokens and clean up spaces before punctuation.
package unigram

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// unkPenalty is how much lower than the lowest piece score an unknown
// character scores, as in SentencePiece and Hugging Face.
const unkPenalty = 10.0

// Config controls length and decoding.
type Config struct {
	// MaxLength is the maximum sequence length including the post-processor's
	// special tokens (model_max_length in tokenizer_config.json); 0 means no
	// limit.
	MaxLength int
	// CleanUpSpaces removes spaces before punctuation and in English
	// contractions when decoding (clean_up_tokenization_spaces).
	CleanUpSpaces bool
}

// Tokenizer turns text into ids and back.
type Tokenizer struct {
	config Config

	pieces      []string
	scores      []float64
	vocab       map[string]int
	maxPieceLen int
	unkID       int
	unkScore    float64

	added   []addedToken // longest first, for matching
	special map[int]bool
	content map[int]string // ids of added tokens beyond the vocabulary

	normalizers  []func(string) string
	wsSplit      bool
	meta         *metaspace
	template     []templateItem
	decodeMeta   *metaspace
	templateSize int
}

type addedToken struct {
	id      int
	content string
}

type metaspace struct {
	replacement string
	prepend     string // "always", "first" or "never"
	split       bool
}

type templateItem struct {
	sequence bool
	ids      []int
}

// LoadDir loads tokenizer.json and reads model_max_length and
// clean_up_tokenization_spaces from tokenizer_config.json, if present.
func LoadDir(dir string) (*Tokenizer, error) {
	var config Config
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json"))
	switch {
	case err == nil:
		var tc struct {
			ModelMaxLength float64 `json:"model_max_length"`
			CleanUp        *bool   `json:"clean_up_tokenization_spaces"`
		}
		if err := json.Unmarshal(raw, &tc); err != nil {
			return nil, fmt.Errorf("unigram: parse tokenizer_config.json: %w", err)
		}
		if tc.ModelMaxLength > 0 && tc.ModelMaxLength < 1<<30 {
			config.MaxLength = int(tc.ModelMaxLength)
		}
		config.CleanUpSpaces = tc.CleanUp == nil || *tc.CleanUp // transformers' default is true
	case errors.Is(err, os.ErrNotExist):
		config.CleanUpSpaces = true
	default:
		return nil, fmt.Errorf("unigram: %w", err)
	}
	return Load(filepath.Join(dir, "tokenizer.json"), config)
}

// Load reads a tokenizer.json whose model is Unigram. Pipeline components
// other than the ones T5-style tokenizers use are rejected rather than
// approximated.
func Load(path string, config Config) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("unigram: %w", err)
	}
	var tj struct {
		AddedTokens []struct {
			ID         int    `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
		} `json:"added_tokens"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		Decoder       json.RawMessage `json:"decoder"`
		Model         struct {
			Type         string              `json:"type"`
			Vocab        [][]json.RawMessage `json:"vocab"`
			UnkID        *int                `json:"unk_id"`
			ByteFallback bool                `json:"byte_fallback"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("unigram: parse %s: %w", path, err)
	}
	// Older tokenizer.json files leave out the model type; a Unigram model is
	// then recognized by its unk_id and scored vocabulary.
	if tj.Model.Type != "Unigram" && tj.Model.Type != "" {
		return nil, fmt.Errorf("unigram: %s has a %q model", path, tj.Model.Type)
	}
	if tj.Model.ByteFallback {
		return nil, errors.New("unigram: byte_fallback is not supported")
	}
	if tj.Model.UnkID == nil {
		return nil, errors.New("unigram: the model has no unk_id")
	}

	t := &Tokenizer{config: config, vocab: map[string]int{}, special: map[int]bool{}, content: map[int]string{}, unkID: *tj.Model.UnkID}
	minScore := math.Inf(1)
	for i, entry := range tj.Model.Vocab {
		var piece string
		var score float64
		if len(entry) != 2 || json.Unmarshal(entry[0], &piece) != nil || json.Unmarshal(entry[1], &score) != nil {
			return nil, fmt.Errorf("unigram: vocab entry %d is not [piece, score]", i)
		}
		t.pieces = append(t.pieces, piece)
		t.scores = append(t.scores, score)
		if _, dup := t.vocab[piece]; !dup {
			t.vocab[piece] = i
		}
		t.maxPieceLen = max(t.maxPieceLen, len(piece))
		minScore = min(minScore, score)
	}
	if t.unkID < 0 || t.unkID >= len(t.pieces) {
		return nil, fmt.Errorf("unigram: unk_id %d outside a vocab of %d", t.unkID, len(t.pieces))
	}
	t.unkScore = minScore - unkPenalty

	for _, a := range tj.AddedTokens {
		if a.Normalized {
			return nil, fmt.Errorf("unigram: added token %q is matched after normalization, which is not supported", a.Content)
		}
		t.added = append(t.added, addedToken{id: a.ID, content: a.Content})
		if a.Special {
			t.special[a.ID] = true
		}
		if a.ID >= len(t.pieces) {
			t.content[a.ID] = a.Content
		}
	}

	if err := t.parseNormalizer(tj.Normalizer); err != nil {
		return nil, err
	}
	if err := t.parsePreTokenizer(tj.PreTokenizer); err != nil {
		return nil, err
	}
	if err := t.parsePostProcessor(tj.PostProcessor); err != nil {
		return nil, err
	}
	if err := t.parseDecoder(tj.Decoder); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *Tokenizer) parseNormalizer(raw json.RawMessage) error {
	if isNull(raw) {
		return nil
	}
	var n struct {
		Type        string            `json:"type"`
		Charsmap    string            `json:"precompiled_charsmap"`
		Normalizers []json.RawMessage `json:"normalizers"`
		Pattern     struct {
			String *string `json:"String"`
			Regex  *string `json:"Regex"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &n); err != nil {
		return fmt.Errorf("unigram: parse normalizer: %w", err)
	}
	switch n.Type {
	case "Sequence":
		for _, child := range n.Normalizers {
			if err := t.parseNormalizer(child); err != nil {
				return err
			}
		}
	case "Precompiled":
		blob, err := base64.StdEncoding.DecodeString(n.Charsmap)
		if err != nil {
			return fmt.Errorf("unigram: precompiled_charsmap: %w", err)
		}
		if len(blob) == 0 {
			return nil // an empty map normalizes nothing
		}
		p, err := parsePrecompiled(blob)
		if err != nil {
			return fmt.Errorf("unigram: %w", err)
		}
		t.normalizers = append(t.normalizers, p.normalize)
	case "Replace":
		switch {
		case n.Pattern.String != nil:
			old, repl := *n.Pattern.String, n.Content
			t.normalizers = append(t.normalizers, func(s string) string { return strings.ReplaceAll(s, old, repl) })
		case n.Pattern.Regex != nil:
			re, err := regexp.Compile(*n.Pattern.Regex)
			if err != nil {
				return fmt.Errorf("unigram: Replace pattern %q: %w", *n.Pattern.Regex, err)
			}
			repl := n.Content
			t.normalizers = append(t.normalizers, func(s string) string { return re.ReplaceAllLiteralString(s, repl) })
		default:
			return errors.New("unigram: Replace normalizer without a pattern")
		}
	default:
		return fmt.Errorf("unigram: normalizer %q is not supported", n.Type)
	}
	return nil
}

func (t *Tokenizer) parsePreTokenizer(raw json.RawMessage) error {
	if isNull(raw) {
		return nil
	}
	var p struct {
		Type          string            `json:"type"`
		PreTokenizers []json.RawMessage `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("unigram: parse pre_tokenizer: %w", err)
	}
	switch p.Type {
	case "Sequence":
		for _, child := range p.PreTokenizers {
			if err := t.parsePreTokenizer(child); err != nil {
				return err
			}
		}
	case "WhitespaceSplit":
		if t.meta != nil {
			return errors.New("unigram: WhitespaceSplit after Metaspace is not supported")
		}
		t.wsSplit = true
	case "Metaspace":
		m, err := parseMetaspace(raw)
		if err != nil {
			return err
		}
		t.meta = m
	default:
		return fmt.Errorf("unigram: pre_tokenizer %q is not supported", p.Type)
	}
	return nil
}

func (t *Tokenizer) parsePostProcessor(raw json.RawMessage) error {
	if isNull(raw) {
		return nil
	}
	var p struct {
		Type   string `json:"type"`
		Single []struct {
			Sequence *struct {
				ID string `json:"id"`
			} `json:"Sequence"`
			SpecialToken *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
		} `json:"single"`
		SpecialTokens map[string]struct {
			IDs []int `json:"ids"`
		} `json:"special_tokens"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("unigram: parse post_processor: %w", err)
	}
	if p.Type != "TemplateProcessing" {
		return fmt.Errorf("unigram: post_processor %q is not supported", p.Type)
	}
	for _, item := range p.Single {
		switch {
		case item.Sequence != nil:
			t.template = append(t.template, templateItem{sequence: true})
		case item.SpecialToken != nil:
			st, ok := p.SpecialTokens[item.SpecialToken.ID]
			if !ok {
				return fmt.Errorf("unigram: template token %q has no ids", item.SpecialToken.ID)
			}
			t.template = append(t.template, templateItem{ids: st.IDs})
			t.templateSize += len(st.IDs)
		}
	}
	return nil
}

func (t *Tokenizer) parseDecoder(raw json.RawMessage) error {
	if isNull(raw) {
		return nil
	}
	var d struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return fmt.Errorf("unigram: parse decoder: %w", err)
	}
	if d.Type != "Metaspace" {
		return fmt.Errorf("unigram: decoder %q is not supported", d.Type)
	}
	m, err := parseMetaspace(raw)
	if err != nil {
		return err
	}
	t.decodeMeta = m
	return nil
}

// parseMetaspace reads both the current (prepend_scheme) and the older
// (add_prefix_space) spellings.
func parseMetaspace(raw json.RawMessage) (*metaspace, error) {
	var m struct {
		Replacement    string `json:"replacement"`
		PrependScheme  string `json:"prepend_scheme"`
		AddPrefixSpace *bool  `json:"add_prefix_space"`
		Split          *bool  `json:"split"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("unigram: parse Metaspace: %w", err)
	}
	ms := &metaspace{replacement: m.Replacement, prepend: m.PrependScheme, split: m.Split == nil || *m.Split}
	if ms.replacement == "" {
		ms.replacement = "▁"
	}
	if ms.prepend == "" {
		ms.prepend = "always"
		if m.AddPrefixSpace != nil && !*m.AddPrefixSpace {
			ms.prepend = "never"
		}
	}
	switch ms.prepend {
	case "always", "first", "never":
	default:
		return nil, fmt.Errorf("unigram: Metaspace prepend_scheme %q is not supported", ms.prepend)
	}
	return ms, nil
}

func isNull(raw json.RawMessage) bool {
	return len(raw) == 0 || string(raw) == "null"
}

// Encode returns the ids for text, including the post-processor's tokens.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	first := true
	for text != "" {
		pos, tok := t.nextAdded(text)
		if pos > 0 {
			ids = append(ids, t.encodeText(text[:pos], first)...)
		}
		if tok == nil {
			break
		}
		ids = append(ids, tok.id)
		first = false
		text = text[pos+len(tok.content):]
	}
	if limit := t.config.MaxLength - t.templateSize; t.config.MaxLength > 0 && len(ids) > limit {
		ids = ids[:max(limit, 0)]
	}
	if t.template == nil {
		return ids
	}
	var out []int
	for _, item := range t.template {
		if item.sequence {
			out = append(out, ids...)
		} else {
			out = append(out, item.ids...)
		}
	}
	return out
}

// nextAdded finds the leftmost added token in text, preferring the longest
// at a position. It returns len(text) and nil when there is none.
func (t *Tokenizer) nextAdded(text string) (int, *addedToken) {
	best, bestTok := len(text), (*addedToken)(nil)
	for i := range t.added {
		a := &t.added[i]
		j := strings.Index(text, a.content)
		if j >= 0 && (j < best || (j == best && len(a.content) > len(bestTok.content))) {
			best, bestTok = j, a
		}
	}
	return best, bestTok
}

// encodeText runs normalization, pre-tokenization and segmentation on a
// stretch of input with no added tokens in it. atStart says it begins the
// input, for Metaspace's "first" scheme.
func (t *Tokenizer) encodeText(text string, atStart bool) []int {
	for _, n := range t.normalizers {
		text = n(text)
	}
	words := []string{text}
	if t.wsSplit {
		words = strings.FieldsFunc(text, unicode.IsSpace)
	}
	var ids []int
	for i, w := range words {
		if t.meta != nil {
			for _, piece := range t.meta.apply(w, atStart && i == 0) {
				ids = append(ids, t.segment(piece)...)
			}
			continue
		}
		ids = append(ids, t.segment(w)...)
	}
	return ids
}

// apply replaces spaces with the replacement, prepends it per the scheme,
// and splits before each replacement, keeping it at the start of a piece.
func (m *metaspace) apply(w string, first bool) []string {
	w = strings.ReplaceAll(w, " ", m.replacement)
	if (m.prepend == "always" || (m.prepend == "first" && first)) && !strings.HasPrefix(w, m.replacement) {
		w = m.replacement + w
	}
	if !m.split {
		return []string{w}
	}
	var pieces []string
	for w != "" {
		next := strings.Index(w[len(m.replacement):], m.replacement)
		if !strings.HasPrefix(w, m.replacement) {
			next = strings.Index(w, m.replacement)
		} else if next >= 0 {
			next += len(m.replacement)
		}
		if next <= 0 {
			pieces = append(pieces, w)
			break
		}
		pieces = append(pieces, w[:next])
		w = w[next:]
	}
	return pieces
}

// segment returns the best-scoring sequence of pieces covering s.
func (t *Tokenizer) segment(s string) []int {
	if s == "" {
		return nil
	}
	// A lattice over byte offsets at character boundaries, as Hugging Face
	// builds it: nodes in insertion order, which breaks ties the same way.
	type node struct {
		id         int
		start, end int
		score      float64
		total      float64
		prev       int // index into nodes, -1 for the start
	}
	var nodes []node
	endsAt := make([][]int, len(s)+1)
	for pos := 0; pos < len(s); {
		_, size := utf8.DecodeRuneInString(s[pos:])
		single := false
		for l := 1; l <= t.maxPieceLen && pos+l <= len(s); l++ {
			if id, ok := t.vocab[s[pos:pos+l]]; ok {
				nodes = append(nodes, node{id: id, start: pos, end: pos + l, score: t.scores[id]})
				endsAt[pos+l] = append(endsAt[pos+l], len(nodes)-1)
				single = single || l == size
			}
		}
		if !single {
			nodes = append(nodes, node{id: t.unkID, start: pos, end: pos + size, score: t.unkScore})
			endsAt[pos+size] = append(endsAt[pos+size], len(nodes)-1)
		}
		pos += size
	}
	// Nodes were added in order of start position, so each node's
	// predecessors (nodes ending at its start) are final when it is reached.
	for i := range nodes {
		n := &nodes[i]
		n.prev = -1
		if n.start == 0 {
			n.total = n.score
			continue
		}
		bestTotal, best := 0.0, -1
		for _, j := range endsAt[n.start] {
			if total := nodes[j].total + n.score; best < 0 || total > bestTotal {
				bestTotal, best = total, j
			}
		}
		if best < 0 {
			n.total = math.Inf(-1) // unreachable
			continue
		}
		n.total, n.prev = bestTotal, best
	}
	bestTotal, last := 0.0, -1
	for _, j := range endsAt[len(s)] {
		if last < 0 || nodes[j].total > bestTotal {
			bestTotal, last = nodes[j].total, j
		}
	}
	var reversed []int
	for j := last; j >= 0; j = nodes[j].prev {
		reversed = append(reversed, nodes[j].id)
	}
	ids := make([]int, 0, len(reversed))
	for i := len(reversed) - 1; i >= 0; i-- {
		id := reversed[i]
		if id == t.unkID && len(ids) > 0 && ids[len(ids)-1] == t.unkID {
			continue // fuse consecutive unknown characters into one <unk>
		}
		ids = append(ids, id)
	}
	return ids
}

// Decode turns ids back into text, skipping special tokens if asked.
func (t *Tokenizer) Decode(ids []int, skipSpecial bool) string {
	var b strings.Builder
	first := true
	for _, id := range ids {
		if skipSpecial && t.special[id] {
			continue
		}
		piece, ok := t.content[id]
		if !ok {
			if id < 0 || id >= len(t.pieces) {
				continue
			}
			piece = t.pieces[id]
		}
		if m := t.decodeMeta; m != nil {
			piece = strings.ReplaceAll(piece, m.replacement, " ")
			if first && m.prepend != "never" {
				piece = strings.TrimPrefix(piece, " ")
			}
		}
		b.WriteString(piece)
		first = false
	}
	out := b.String()
	if t.config.CleanUpSpaces {
		out = cleanUp(out)
	}
	return out
}

// cleanUp is transformers' clean_up_tokenization.
func cleanUp(s string) string {
	for _, r := range [][2]string{
		{" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"},
		{" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"},
	} {
		s = strings.ReplaceAll(s, r[0], r[1])
	}
	return s
}

// VocabSize returns the number of ids, including added tokens past the
// vocabulary.
func (t *Tokenizer) VocabSize() int {
	n := len(t.pieces)
	for id := range t.content {
		n = max(n, id+1)
	}
	return n
}
