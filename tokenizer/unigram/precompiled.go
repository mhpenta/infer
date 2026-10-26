package unigram

import (
	"encoding/binary"
	"fmt"
	"strings"

	"github.com/rivo/uniseg"
)

// precompiled is SentencePiece's compiled normalization rule (for T5,
// NFKC-style folding: full-width forms to ASCII, ligatures split, compatibility
// characters replaced). It is a double-array trie (darts-clone) mapping input
// byte sequences to offsets into a blob of NUL-terminated replacements.
type precompiled struct {
	trie        []uint32
	replacement string
}

// parsePrecompiled reads the precompiled_charsmap blob: a little-endian
// uint32 trie size in bytes, the trie's uint32 units, then the replacements.
func parsePrecompiled(blob []byte) (*precompiled, error) {
	if len(blob) < 4 {
		return nil, fmt.Errorf("precompiled charsmap is %d bytes", len(blob))
	}
	size := int(binary.LittleEndian.Uint32(blob))
	if size%4 != 0 || 4+size > len(blob) {
		return nil, fmt.Errorf("precompiled charsmap trie size %d does not fit %d bytes", size, len(blob))
	}
	p := &precompiled{trie: make([]uint32, size/4), replacement: string(blob[4+size:])}
	for i := range p.trie {
		p.trie[i] = binary.LittleEndian.Uint32(blob[4+4*i:])
	}
	return p, nil
}

// prefixMatch returns the value of the shortest prefix of key stored in the
// trie, the first result of darts-clone's common prefix search.
func (p *precompiled) prefixMatch(key string) (uint32, bool) {
	unit := func(pos uint32) (uint32, bool) {
		if int(pos) >= len(p.trie) {
			return 0, false
		}
		return p.trie[pos], true
	}
	offset := func(u uint32) uint32 { return (u >> 10) << ((u & (1 << 9)) >> 6) }
	pos := uint32(0)
	u, ok := unit(pos)
	if !ok {
		return 0, false
	}
	pos ^= offset(u)
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c == 0 {
			break
		}
		pos ^= uint32(c)
		if u, ok = unit(pos); !ok || u&((1<<31)|0xff) != uint32(c) {
			return 0, false
		}
		pos ^= offset(u)
		if (u>>8)&1 == 1 {
			leaf, ok := unit(pos)
			if !ok {
				return 0, false
			}
			return leaf & ((1 << 31) - 1), true
		}
	}
	return 0, false
}

// transform returns the replacement for chunk if any prefix of it is in the
// trie. As in Hugging Face's spm_precompiled, the whole chunk is replaced
// even when only a prefix matched.
func (p *precompiled) transform(chunk string) (string, bool) {
	v, ok := p.prefixMatch(chunk)
	if !ok || int(v) > len(p.replacement) {
		return "", false
	}
	rest := p.replacement[v:]
	if end := strings.IndexByte(rest, 0); end >= 0 {
		rest = rest[:end]
	}
	return rest, true
}

// normalize applies the rule the way Hugging Face's Precompiled normalizer
// does: whole grapheme clusters (under 6 bytes) are looked up first, so a
// base letter and its combining marks can map to one composed character;
// otherwise each character of the cluster is looked up on its own.
func (p *precompiled) normalize(s string) string {
	var b strings.Builder
	g := uniseg.NewGraphemes(s)
	for g.Next() {
		cluster := g.Str()
		if len(cluster) < 6 {
			if r, ok := p.transform(cluster); ok {
				b.WriteString(r)
				continue
			}
		}
		for _, c := range cluster {
			ch := string(c)
			if r, ok := p.transform(ch); ok {
				b.WriteString(r)
			} else {
				b.WriteString(ch)
			}
		}
	}
	return b.String()
}
