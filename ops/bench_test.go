package ops

import (
	"testing"
	"time"
)

func TestDiagnostics(t *testing.T) {
	peak := PeakGFLOPS(20 * time.Millisecond)
	kern := KernelGFLOPS(20 * time.Millisecond)
	mm := MatMulGFLOPS(20*time.Millisecond, 65, 384, 384, "f32")
	halfRate := MatMulGFLOPS(20*time.Millisecond, 4, 1024, 1024, "f16")

	t.Logf("%s: peak %.1f, kernels %v, matmul 65x384x384 %.1f GFLOP/s", Kernel(), peak, kern, mm)
	t.Logf("fp16 panels 4x1024x1024 %.1f GFLOP/s", halfRate)
	if kern["6x16"] <= 0 || mm <= 0 || halfRate <= 0 {
		t.Fatal("kernel and matmul rates must be positive")
	}
	if Kernel() != "go" && peak <= 0 {
		t.Fatal("SIMD build reported no peak rate")
	}
}
