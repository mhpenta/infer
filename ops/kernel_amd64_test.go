//go:build amd64 && !purego

package ops

import (
	"os"
	"testing"
)

// TestAVX2Active fails when INFER_REQUIRE_AVX2=1 and the CPU lacks AVX2 or
// FMA, so a CI run cannot silently test only the portable fallback.
func TestAVX2Active(t *testing.T) {
	if !useAVX2 && os.Getenv("INFER_REQUIRE_AVX2") == "1" {
		t.Fatal("AVX2/FMA not detected")
	}
	t.Logf("kernel: %s", Kernel())
}
