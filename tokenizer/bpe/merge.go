package bpe

import "container/heap"

// merge is the result of merging a pair of tokens: the merged token's id
// and the merge's rank (its line in the merges list; lower merges first).
type merge struct {
	rank, id int32
}

// pairKey packs two token ids into one map key.
func pairKey(a, b int32) uint64 { return uint64(uint32(a))<<32 | uint64(uint32(b)) }

// symbol is one token of a word being merged, in a doubly linked list.
type symbol struct {
	id         int32
	prev, next int32 // indexes into the word, -1 at the ends
	gone       bool  // merged into its left neighbor
}

// candidate is a possible merge of the symbol at pos with the one after it.
type candidate struct {
	rank, pos, id int32
}

// candidates is a min-heap by rank, then position: the order in which the
// tokenizers library applies merges.
type candidates []candidate

func (c candidates) Len() int { return len(c) }
func (c candidates) Less(i, j int) bool {
	if c[i].rank != c[j].rank {
		return c[i].rank < c[j].rank
	}
	return c[i].pos < c[j].pos
}
func (c candidates) Swap(i, j int) { c[i], c[j] = c[j], c[i] }
func (c *candidates) Push(x any)   { *c = append(*c, x.(candidate)) }
func (c *candidates) Pop() any {
	old := *c
	x := old[len(old)-1]
	*c = old[:len(old)-1]
	return x
}

// mergeWord applies merges to ids (the word's byte tokens) until none
// applies, always taking the lowest-ranked pair present and the leftmost of
// equal ones, and appends the result to dst.
func (t *Tokenizer) mergeWord(dst []int, ids []int32) []int {
	if len(ids) == 1 {
		return append(dst, int(ids[0]))
	}
	syms := make([]symbol, len(ids))
	var queue candidates
	for i, id := range ids {
		syms[i] = symbol{id: id, prev: int32(i - 1), next: int32(i + 1)}
		if i > 0 {
			if m, ok := t.merges[pairKey(ids[i-1], id)]; ok {
				queue = append(queue, candidate{rank: m.rank, pos: int32(i - 1), id: m.id})
			}
		}
	}
	syms[len(syms)-1].next = -1
	heap.Init(&queue)
	for queue.Len() > 0 {
		c := heap.Pop(&queue).(candidate)
		left := &syms[c.pos]
		if left.gone || left.next < 0 {
			continue
		}
		right := &syms[left.next]
		// Skip entries made stale by an earlier merge of either symbol.
		if m, ok := t.merges[pairKey(left.id, right.id)]; !ok || m.id != c.id {
			continue
		}
		left.id, left.next = c.id, right.next
		right.gone = true
		if left.next >= 0 {
			syms[left.next].prev = c.pos
		}
		if left.prev >= 0 {
			if m, ok := t.merges[pairKey(syms[left.prev].id, left.id)]; ok {
				heap.Push(&queue, candidate{rank: m.rank, pos: left.prev, id: m.id})
			}
		}
		if left.next >= 0 {
			if m, ok := t.merges[pairKey(left.id, syms[left.next].id)]; ok {
				heap.Push(&queue, candidate{rank: m.rank, pos: c.pos, id: m.id})
			}
		}
	}
	for i := int32(0); i >= 0; i = syms[i].next {
		dst = append(dst, int(syms[i].id))
	}
	return dst
}
