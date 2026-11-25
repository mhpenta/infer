package bpe

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// splitFunc cuts s into pieces, appending them to dst. Like the tokenizers
// library's Split pre-tokenizer with "Isolated" behavior, every match is a
// piece and so is every stretch between matches.
type splitFunc func(dst []string, s string) []string

// Split patterns this package implements. Go's regexp (RE2) cannot run them
// as written: it has no lookahead, and its \s is ASCII-only where the
// tokenizers library's Oniguruma \s is Unicode White_Space. So each pattern
// has a hand-written scanner, keyed by its exact source.
var splitPatterns = map[string]splitFunc{
	`\p{N}{1,3}`: splitDigits(3),
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:            splitWords(wordStyle{numberRuns: true}), // MiniCPM5
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:             splitWords(wordStyle{}),                 // Qwen2, Qwen3
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`: splitWords(wordStyle{marks: true}),      // Qwen3.5
	o200kPattern: splitO200k, // GPT-4o (o200k_base), Inkling 2
}

// o200kPattern is o200k_base's split pattern, as tokenizer.json writes it.
const o200kPattern = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// splitDigits isolates runs of up to limit number characters (\p{N}), taken
// from the left, so "12345" becomes "123", "45".
func splitDigits(limit int) splitFunc {
	return func(dst []string, s string) []string {
		gap := 0 // start of the text since the last match
		for i := 0; i < len(s); {
			r, size := utf8.DecodeRuneInString(s[i:])
			if !unicode.IsNumber(r) {
				i += size
				continue
			}
			end := i
			for n := 0; n < limit && end < len(s); n++ {
				r, size := utf8.DecodeRuneInString(s[end:])
				if !unicode.IsNumber(r) {
					break
				}
				end += size
			}
			if i > gap {
				dst = append(dst, s[gap:i])
			}
			dst = append(dst, s[i:end])
			i, gap = end, end
		}
		if gap < len(s) {
			dst = append(dst, s[gap:])
		}
		return dst
	}
}

// wordStyle selects a variant of splitWords's pattern.
type wordStyle struct {
	// numberRuns makes a number \p{N}+, a run, instead of \p{N}, one number
	// character per piece as Qwen's tokenizers write it.
	numberRuns bool
	// marks makes a word [\p{L}\p{M}]+ and punctuation [^\s\p{L}\p{M}\p{N}],
	// keeping combining marks in their words, as Qwen3.5's tokenizer writes
	// them.
	marks bool
}

// splitWords returns the GPT-4-style word split, cl100k's pattern in the
// variant style picks:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)   contractions
//	[^\r\n\p{L}\p{N}]?\p{L}+       a word, with one leading non-letter
//	\p{N}+ or \p{N}                a number
//	 ?[^\s\p{L}\p{N}]+[\r\n]*      punctuation, with a leading space and
//	                               trailing newlines
//	\s*[\r\n]+                     whitespace through the last newline
//	\s+(?!\S)                      whitespace, leaving the last character to
//	                               the next word
//	\s+                            whitespace
//
// Every character matches some alternative, so the pieces are exactly the
// successive matches.
func splitWords(style wordStyle) splitFunc {
	return func(dst []string, s string) []string {
		for i := 0; i < len(s); {
			end := wordEnd(s, i, style)
			dst = append(dst, s[i:end])
			i = end
		}
		return dst
	}
}

// wordEnd returns the end of the match at s[i:], trying the alternatives in
// order as a backtracking regex engine does.
func wordEnd(s string, i int, style wordStyle) int {
	r0, size0 := utf8.DecodeRuneInString(s[i:])

	if r0 == '\'' {
		if n := contraction(s[i+1:]); n > 0 {
			return i + 1 + n
		}
	}

	// A word's optional leading character, [^\r\n\p{L}\p{N}], can be a mark;
	// when marks are word characters, the run then simply starts there.
	if style.isWord(r0) {
		return style.wordRun(s, i)
	}
	if r0 != '\r' && r0 != '\n' && !isNumber(r0) && i+size0 < len(s) {
		if r1, _ := utf8.DecodeRuneInString(s[i+size0:]); style.isWord(r1) {
			return style.wordRun(s, i+size0)
		}
	}

	if isNumber(r0) && !style.numberRuns {
		return i + size0
	}
	if isNumber(r0) {
		end := i
		for end < len(s) {
			r, size := utf8.DecodeRuneInString(s[end:])
			if !isNumber(r) {
				break
			}
			end += size
		}
		return end
	}

	if p := i; style.isPunct(r0) || (r0 == ' ' && p+1 < len(s) && style.isPunct(firstRune(s[p+1:]))) {
		if r0 == ' ' {
			p++
		}
		for p < len(s) {
			r, size := utf8.DecodeRuneInString(s[p:])
			if !style.isPunct(r) {
				break
			}
			p += size
		}
		for p < len(s) && (s[p] == '\r' || s[p] == '\n') {
			p++
		}
		return p
	}

	// r0 is whitespace.
	return whitespaceEnd(s, i)
}

// whitespaceEnd returns the end of the whitespace alternatives at s[i:],
// \s*[\r\n]+, \s+(?!\S) and \s+, where s[i] is whitespace.
func whitespaceEnd(s string, i int) int {
	// Find the run and its last newline.
	end, lastNewline, lastSize := i, -1, 0
	for end < len(s) {
		r, size := utf8.DecodeRuneInString(s[end:])
		if !unicode.IsSpace(r) {
			break
		}
		if r == '\r' || r == '\n' {
			lastNewline = end
		}
		end, lastSize = end+size, size
	}
	switch {
	case lastNewline >= 0: // \s*[\r\n]+
		return lastNewline + 1
	case end == len(s): // \s+(?!\S) at the end of the text
		return end
	case end-lastSize > i: // \s+(?!\S), backing off one character
		return end - lastSize
	}
	return end // \s+
}

// contraction returns the length of a case-insensitive s, t, re, ve, m, ll
// or d at the start of rest (after an apostrophe), or 0. Case folding also
// matches ſ (U+017F, long s) for s, as Unicode simple folding does.
func contraction(rest string) int {
	for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
		if n := hasFoldPrefix(rest, c); n > 0 {
			return n
		}
	}
	return 0
}

// hasFoldPrefix returns the byte length of a case-insensitive match of the
// lowercase ASCII word at the start of s, or 0.
func hasFoldPrefix(s, word string) int {
	n := 0
	for _, c := range []byte(word) {
		switch {
		case n < len(s) && (s[n] == c || s[n] == c-'a'+'A'):
			n++
		case c == 's' && strings.HasPrefix(s[n:], "ſ"):
			n += len("ſ")
		default:
			return 0
		}
	}
	return n
}

// wordRun returns the end of the run of word characters starting at s[i:].
func (w wordStyle) wordRun(s string, i int) int {
	for i < len(s) {
		r, size := utf8.DecodeRuneInString(s[i:])
		if !w.isWord(r) {
			break
		}
		i += size
	}
	return i
}

func firstRune(s string) rune {
	r, _ := utf8.DecodeRuneInString(s)
	return r
}

func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }

// isWord is \p{L}, or [\p{L}\p{M}] with marks.
func (w wordStyle) isWord(r rune) bool { return isLetter(r) || w.marks && unicode.IsMark(r) }

// isPunct is [^\s\p{L}\p{N}], or [^\s\p{L}\p{M}\p{N}] with marks.
func (w wordStyle) isPunct(r rune) bool { return !unicode.IsSpace(r) && !w.isWord(r) && !isNumber(r) }

// byteChars is GPT-2's bytes_to_unicode: printable Latin-1 bytes stand for
// themselves, and the rest map to code points from U+0100 up, so every byte
// has a visible character.
var byteChars, charBytes = func() ([256]rune, map[rune]byte) {
	var chars [256]rune
	bytes := map[rune]byte{}
	next := rune(256)
	for b := range 256 {
		r := rune(b)
		if !(('!' <= r && r <= '~') || ('¡' <= r && r <= '¬') || ('®' <= r && r <= 'ÿ')) {
			r = next
			next++
		}
		chars[b] = r
		bytes[r] = byte(b)
	}
	return chars, bytes
}()

// splitO200k is o200k_base's split:
//
//	P?U*L+C?          a word ending in lowercase: P is one optional
//	                  [^\r\n\p{L}\p{N}], U is [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}],
//	                  L is [\p{Ll}\p{Lm}\p{Lo}\p{M}], and C an optional
//	                  contraction ('s, 't, 're, 've, 'm, 'll, 'd, any case)
//	P?U+L*C?          a word in capitals
//	\p{N}{1,3}        up to three number characters
//	 ?[^\s\p{L}\p{N}]+[\r\n/]*
//	                  punctuation, with a leading space and trailing
//	                  newlines or slashes
//	\s*[\r\n]+ | \s+(?!\S) | \s+
//	                  whitespace, as in splitWords
//
// Every character matches some alternative, so the pieces are exactly the
// successive matches.
func splitO200k(dst []string, s string) []string {
	for i := 0; i < len(s); {
		end := o200kEnd(s, i)
		dst = append(dst, s[i:end])
		i = end
	}
	return dst
}

func o200kEnd(s string, i int) int {
	if end, ok := o200kWord(s, i, true); ok {
		return end
	}
	if end, ok := o200kWord(s, i, false); ok {
		return end
	}
	r0, size0 := utf8.DecodeRuneInString(s[i:])
	if isNumber(r0) {
		end := i
		for n := 0; n < 3 && end < len(s); n++ {
			r, size := utf8.DecodeRuneInString(s[end:])
			if !isNumber(r) {
				break
			}
			end += size
		}
		return end
	}
	punct := func(r rune) bool { return !unicode.IsSpace(r) && !isLetter(r) && !isNumber(r) }
	if p := i; punct(r0) || (r0 == ' ' && p+size0 < len(s) && punct(firstRune(s[p+size0:]))) {
		if r0 == ' ' {
			p++
		}
		for p < len(s) {
			r, size := utf8.DecodeRuneInString(s[p:])
			if !punct(r) {
				break
			}
			p += size
		}
		for p < len(s) && (s[p] == '\r' || s[p] == '\n' || s[p] == '/') {
			p++
		}
		return p
	}
	return whitespaceEnd(s, i)
}

// o200kWord matches P?U*L+C? (lower) or P?U+L*C? at s[i:], trying the
// choices in the order a backtracking engine does: the optional P taken
// first, then the longest U run, giving back characters one at a time.
func o200kWord(s string, i int, lower bool) (int, bool) {
	r0, size0 := utf8.DecodeRuneInString(s[i:])
	starts := []int{i}
	if r0 != '\r' && r0 != '\n' && !isLetter(r0) && !isNumber(r0) {
		starts = []int{i + size0, i}
	}
	for _, start := range starts {
		ends := []int{start} // ends[k] is the end of a U run of k characters
		for j := start; j < len(s); {
			r, size := utf8.DecodeRuneInString(s[j:])
			if !o200kUpper(r) {
				break
			}
			j += size
			ends = append(ends, j)
		}
		if !lower {
			if len(ends) < 2 {
				continue
			}
			return withContraction(s, runEnd(s, ends[len(ends)-1], o200kLower)), true
		}
		for k := len(ends) - 1; k >= 0; k-- {
			if e := runEnd(s, ends[k], o200kLower); e > ends[k] {
				return withContraction(s, e), true
			}
		}
	}
	return 0, false
}

func o200kUpper(r rune) bool {
	return unicode.In(r, unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M)
}

func o200kLower(r rune) bool {
	return unicode.In(r, unicode.Ll, unicode.Lm, unicode.Lo, unicode.M)
}

// runEnd returns the end of the run of characters in class from s[i:].
func runEnd(s string, i int, class func(rune) bool) int {
	for i < len(s) {
		r, size := utf8.DecodeRuneInString(s[i:])
		if !class(r) {
			break
		}
		i += size
	}
	return i
}

// withContraction extends a word ending at end by a contraction after it.
func withContraction(s string, end int) int {
	if end < len(s) && s[end] == '\'' {
		if n := contraction(s[end+1:]); n > 0 {
			return end + 1 + n
		}
	}
	return end
}
