// Package accel chooses where a model runs: on the CPU, or on an
// accelerator this build and machine have. Today the one accelerator is the
// Apple silicon GPU, through package metal.
//
// A model's loader calls Resolve with the device its options ask for, and,
// for Metal, OpenMetal, which opens the GPU and compiles and self-checks the
// kernels once per process. The loader decides the rest, such as whether its
// weights suit the GPU and fit in its memory, because only it knows them.
package accel

import (
	"fmt"
	"os"
	"sync"

	"github.com/mhpenta/infer/metal"
)

// Device names.
const (
	CPU   = "cpu"
	Metal = "metal"
	// Auto picks Metal when it is available and suits the model, and the
	// CPU otherwise.
	Auto = "auto"
)

// EnvVar names the environment variable that sets the device when a model's
// options leave it empty.
const EnvVar = "INFER_DEVICE"

// Resolve returns the device asked for: pref if it is not empty, else
// $INFER_DEVICE, else Auto.
func Resolve(pref string) (string, error) {
	if pref == "" {
		pref = os.Getenv(EnvVar)
	}
	switch pref {
	case "":
		return Auto, nil
	case CPU, Metal, Auto:
		return pref, nil
	}
	return "", fmt.Errorf("unknown device %q (want %s, %s or %s)", pref, CPU, Metal, Auto)
}

// GPU is the process's Metal device and its kernels, shared by every model.
// Both are safe for concurrent use.
type GPU struct {
	Device  *metal.Device
	Kernels *metal.Kernels
}

var gpu struct {
	once sync.Once
	gpu  *GPU
	err  error
}

// OpenMetal returns the shared GPU, opening it, compiling the kernels and
// running their self-check (Check) on first use. The error wraps
// metal.ErrUnavailable when there is no Metal device; a failed compilation
// or self-check is an error too, since it means the OS's shader compiler or
// driver computes something the tests did not see.
func OpenMetal() (*GPU, error) {
	gpu.once.Do(func() {
		d, err := metal.Open()
		if err != nil {
			gpu.err = err
			return
		}
		k, err := d.Kernels()
		if err != nil {
			d.Close()
			gpu.err = err
			return
		}
		if err := Check(d, k); err != nil {
			k.Release()
			d.Close()
			gpu.err = fmt.Errorf("accel: Metal kernels failed their self-check on %s: %w", d.Name(), err)
			return
		}
		gpu.gpu = &GPU{d, k}
	})
	return gpu.gpu, gpu.err
}
