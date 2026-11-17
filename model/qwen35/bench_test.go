package qwen35

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkClassify times classifying the golden cases' premise-hypothesis
// pairs: 22, 94 and 2,575 tokens, reporting tokens per second.
func BenchmarkClassify(b *testing.B) {
	m, g, _ := load(b)
	for _, i := range []int{0, 2, longCase} {
		ids := g.Cases[i].InputIDs
		b.Run(fmt.Sprintf("%dtokens", len(ids)), func(b *testing.B) {
			start := time.Now()
			for range b.N {
				if _, err := m.Classify(ids); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N*len(ids))/time.Since(start).Seconds(), "tokens/s")
		})
	}
}

// BenchmarkClassifyShared times a decision's options from their shared
// prefix, against running each option alone.
func BenchmarkClassifyShared(b *testing.B) {
	m, g, _ := load(b)
	var ids [][]int
	for _, o := range g.Decisions[2].Options {
		ids = append(ids, o.InputIDs)
	}
	n := commonPrefix(ids)
	var suffixes [][]int
	for _, s := range ids {
		suffixes = append(suffixes, s[n:])
	}
	b.Run("shared", func(b *testing.B) {
		for range b.N {
			if _, err := m.ClassifyShared(ids[0][:n], suffixes); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("alone", func(b *testing.B) {
		for range b.N {
			for _, s := range ids {
				if _, err := m.Classify(s); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
}
