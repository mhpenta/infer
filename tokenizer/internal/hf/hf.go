// Package hf holds the parts of Hugging Face's tokenizer.json pipeline that
// every tokenizer model shares: added tokens, the TemplateProcessing
// post-processor and transformers' decoding clean-up.
package hf

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// IsNull reports whether a tokenizer.json component is absent.
func IsNull(raw json.RawMessage) bool {
	return len(raw) == 0 || string(raw) == "null"
}

// AddedToken is one entry of tokenizer.json's added_tokens.
type AddedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	Special    bool   `json:"special"`
	Normalized bool   `json:"normalized"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
}

// Added finds added tokens in text the way the tokenizers library does:
// tokens matched on the raw text (normalized false) first, then tokens
// matched after normalization in the stretches between them, each pass
// taking the leftmost match and, at one position, the longest token.
type Added struct {
	raw, normalized matcher
}

// NewAdded indexes tokens for matching. Options that change how a match
// absorbs the text around it (lstrip, rstrip, single_word) are not
// supported.
func NewAdded(tokens []AddedToken) (*Added, error) {
	a := &Added{raw: matcher{}, normalized: matcher{}}
	for _, t := range tokens {
		if t.LStrip || t.RStrip || t.SingleWord {
			return nil, fmt.Errorf("added token %q: lstrip, rstrip and single_word are not supported", t.Content)
		}
		if t.Content == "" {
			return nil, errors.New("added token with empty content")
		}
		m := a.raw
		if t.Normalized {
			m = a.normalized
		}
		m[t.Content[0]] = append(m[t.Content[0]], t)
	}
	for _, m := range []matcher{a.raw, a.normalized} {
		for b := range m {
			slices.SortStableFunc(m[b], func(x, y AddedToken) int { return cmp.Compare(len(y.Content), len(x.Content)) })
		}
	}
	return a, nil
}

// HasNormalized reports whether some tokens are matched after
// normalization.
func (a *Added) HasNormalized() bool { return len(a.normalized) > 0 }

// matcher holds tokens by first byte, longest first.
type matcher map[byte][]AddedToken

// next returns the position and token of the leftmost match in text, or
// len(text) and nil.
func (m matcher) next(text string) (int, *AddedToken) {
	if len(m) == 0 {
		return len(text), nil
	}
	for i := 0; i < len(text); i++ {
		for j := range m[text[i]] {
			if t := &m[text[i]][j]; strings.HasPrefix(text[i:], t.Content) {
				return i, t
			}
		}
	}
	return len(text), nil
}

// Segment is a stretch of input: an added token (Token non-nil) or text to
// run through the rest of the pipeline.
type Segment struct {
	Text  string
	Token *AddedToken
}

// Split cuts text into added tokens and the text between them. normalize,
// if not nil, is applied to each text stretch before the second pass, as the
// tokenizers library does; the returned text stretches are then normalized.
func (a *Added) Split(text string, normalize func(string) string) []Segment {
	var out []Segment
	for _, s := range split(text, a.raw) {
		if s.Token != nil {
			out = append(out, s)
			continue
		}
		if normalize != nil {
			s.Text = normalize(s.Text)
		}
		out = append(out, split(s.Text, a.normalized)...)
	}
	return out
}

func split(text string, m matcher) []Segment {
	var out []Segment
	for text != "" {
		pos, tok := m.next(text)
		if pos > 0 {
			out = append(out, Segment{Text: text[:pos]})
		}
		if tok == nil {
			break
		}
		out = append(out, Segment{Token: tok})
		text = text[pos+len(tok.Content):]
	}
	return out
}

// Template is a TemplateProcessing post-processor for single sequences.
type Template struct {
	items []templateItem
	size  int // ids the template adds
}

type templateItem struct {
	sequence bool
	ids      []int
}

// ParseTemplate reads a post_processor; a null one adds nothing.
func ParseTemplate(raw json.RawMessage) (Template, error) {
	var t Template
	if IsNull(raw) {
		return t, nil
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
		return t, fmt.Errorf("parse post_processor: %w", err)
	}
	if p.Type != "TemplateProcessing" {
		return t, fmt.Errorf("post_processor %q is not supported", p.Type)
	}
	for _, item := range p.Single {
		switch {
		case item.Sequence != nil:
			t.items = append(t.items, templateItem{sequence: true})
		case item.SpecialToken != nil:
			st, ok := p.SpecialTokens[item.SpecialToken.ID]
			if !ok {
				return t, fmt.Errorf("template token %q has no ids", item.SpecialToken.ID)
			}
			t.items = append(t.items, templateItem{ids: st.IDs})
			t.size += len(st.IDs)
		}
	}
	return t, nil
}

// Size returns how many ids Apply adds.
func (t Template) Size() int { return t.size }

// Apply returns ids wrapped in the template's tokens.
func (t Template) Apply(ids []int) []int {
	if t.items == nil {
		return ids
	}
	out := make([]int, 0, len(ids)+t.size)
	for _, item := range t.items {
		if item.sequence {
			out = append(out, ids...)
		} else {
			out = append(out, item.ids...)
		}
	}
	return out
}

// CleanUp is transformers' clean_up_tokenization: it removes spaces before
// punctuation and inside English contractions.
func CleanUp(s string) string {
	for _, r := range [][2]string{
		{" .", "."}, {" ?", "?"}, {" !", "!"}, {" ,", ","}, {" ' ", "'"},
		{" n't", "n't"}, {" 'm", "'m"}, {" 's", "'s"}, {" 've", "'ve"}, {" 're", "'re"},
	} {
		s = strings.ReplaceAll(s, r[0], r[1])
	}
	return s
}
