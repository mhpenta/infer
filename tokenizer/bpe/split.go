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
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`: splitWords(true),  // MiniCPM5
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`:  splitWords(false), // Qwen2, Qwen3
}

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

// splitWords returns the GPT-4-style word split: cl100k's pattern with
// \p{N}+ when numberRuns is set, or with \p{N}, one number character per
// piece, as Qwen's tokenizers write it:
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
func splitWords(numberRuns bool) splitFunc {
	return func(dst []string, s string) []string {
		for i := 0; i < len(s); {
			end := wordEnd(s, i, numberRuns)
			dst = append(dst, s[i:end])
			i = end
		}
		return dst
	}
}

// wordEnd returns the end of the match at s[i:], trying the alternatives in
// order as a backtracking regex engine does.
func wordEnd(s string, i int, numberRuns bool) int {
	r0, size0 := utf8.DecodeRuneInString(s[i:])

	if r0 == '\'' {
		if n := contraction(s[i+1:]); n > 0 {
			return i + 1 + n
		}
	}

	if isLetter(r0) {
		return letters(s, i)
	}
	if r0 != '\r' && r0 != '\n' && !isNumber(r0) && i+size0 < len(s) {
		if r1, _ := utf8.DecodeRuneInString(s[i+size0:]); isLetter(r1) {
			return letters(s, i+size0)
		}
	}

	if isNumber(r0) && !numberRuns {
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

	if p := i; isPunct(r0) || (r0 == ' ' && p+1 < len(s) && isPunct(firstRune(s[p+1:]))) {
		if r0 == ' ' {
			p++
		}
		for p < len(s) {
			r, size := utf8.DecodeRuneInString(s[p:])
			if !isPunct(r) {
				break
			}
			p += size
		}
		for p < len(s) && (s[p] == '\r' || s[p] == '\n') {
			p++
		}
		return p
	}

	// r0 is whitespace. Find the run and its last newline.
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

// letters returns the end of the run of letters starting at s[i:].
func letters(s string, i int) int {
	for i < len(s) {
		r, size := utf8.DecodeRuneInString(s[i:])
		if !isLetter(r) {
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

// isPunct is [^\s\p{L}\p{N}].
func isPunct(r rune) bool { return !unicode.IsSpace(r) && !isLetter(r) && !isNumber(r) }

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
