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

// TestAVX512SelfCheckPasses fails on a CPU with AVX-512 whose startup
// self-check rejected the AVX-512 kernels, which leaves every format on the
// AVX2 kernels. A too-tight tolerance once did this on every such CPU: int8
// panels at k=301 differ from the portable kernel by 1.4e-3 relative to an
// output that cancels, within float32 rounding of the sums.
func TestAVX512SelfCheckPasses(t *testing.T) {
	if !hasAVX512 {
		t.Skip("no AVX-512")
	}
	if err := checkWideKernel(); err != nil {
		t.Fatal(err)
	}
}
