package accel

import (
	"errors"
	"os"
	"testing"

	"github.com/mhpenta/infer/metal"
)

func TestResolve(t *testing.T) {
	t.Setenv(EnvVar, "")
	for pref, want := range map[string]string{"": Auto, CPU: CPU, Metal: Metal, Auto: Auto} {
		if got, err := Resolve(pref); err != nil || got != want {
			t.Errorf("Resolve(%q) = %q, %v; want %q", pref, got, err, want)
		}
	}
	if _, err := Resolve("cuda"); err == nil {
		t.Error("Resolve accepted an unknown device")
	}
	t.Setenv(EnvVar, CPU)
	if got, _ := Resolve(""); got != CPU {
		t.Errorf("Resolve with %s=cpu = %q", EnvVar, got)
	}
	if got, _ := Resolve(Metal); got != Metal {
		t.Errorf("an explicit device lost to %s: %q", EnvVar, got)
	}
	t.Setenv(EnvVar, "tpu")
	if _, err := Resolve(""); err == nil {
		t.Errorf("Resolve accepted %s=tpu", EnvVar)
	}
}

// TestOpenMetal opens the shared GPU, which runs the self-check, and checks
// the result is shared.
func TestOpenMetal(t *testing.T) {
	g, err := OpenMetal()
	if errors.Is(err, metal.ErrUnavailable) {
		if os.Getenv("INFER_REQUIRE_METAL") == "1" {
			t.Fatal("INFER_REQUIRE_METAL=1 but there is no Metal device")
		}
		t.Skip("no Metal device")
	}
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := OpenMetal(); again != g {
		t.Error("OpenMetal opened a second GPU")
	}
	if err := Check(g.Device, g.Kernels); err != nil {
		t.Fatal(err)
	}
}
