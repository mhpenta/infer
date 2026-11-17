package cuda

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"
)

// vaddPTX is c[i] = a[i] + b[i] for i < n, written by hand. sm_52 is the
// oldest target the current driver accepts; the driver compiles it for
// whatever GPU is present.
const vaddPTX = `
.version 7.0
.target sm_52
.address_size 64

.visible .entry vadd(
	.param .u64 pa,
	.param .u64 pb,
	.param .u64 pc,
	.param .u32 pn
)
{
	.reg .pred  done;
	.reg .b32   i, n, ctaid, ntid, tid;
	.reg .f32   x, y, z;
	.reg .b64   a, b, c, off;

	ld.param.u64    a, [pa];
	ld.param.u64    b, [pb];
	ld.param.u64    c, [pc];
	ld.param.u32    n, [pn];
	mov.u32         ctaid, %ctaid.x;
	mov.u32         ntid, %ntid.x;
	mov.u32         tid, %tid.x;
	mad.lo.u32      i, ctaid, ntid, tid;
	setp.ge.u32     done, i, n;
	@done bra       END;

	cvta.to.global.u64  a, a;
	cvta.to.global.u64  b, b;
	cvta.to.global.u64  c, c;
	mul.wide.u32    off, i, 4;
	add.u64         a, a, off;
	add.u64         b, b, off;
	add.u64         c, c, off;
	ld.global.f32   x, [a];
	ld.global.f32   y, [b];
	add.rn.f32      z, x, y;
	st.global.f32   [c], z;
END:
	ret;
}
`

func openDevice(t *testing.T) *Device {
	t.Helper()
	d, err := Open(0)
	if errors.Is(err, ErrUnavailable) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	return d
}

func loadVadd(t *testing.T, d *Device) *Function {
	t.Helper()
	m, err := d.LoadPTX(vaddPTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.Unload(); err != nil {
			t.Error(err)
		}
	})
	f, err := m.Function("vadd")
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func alloc(t *testing.T, d *Device, size int) *Buffer {
	t.Helper()
	b, err := d.Alloc(size)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Free(); err != nil {
			t.Error(err)
		}
	})
	return b
}

// vectorAdd adds two random vectors of n float32s on the GPU and checks
// the sum bit for bit: add.rn.f32 rounds as Go's float32 addition does. It
// returns an error rather than failing t, so goroutines can call it.
func vectorAdd(d *Device, f *Function, n int, seed uint64) (err error) {
	r := rand.New(rand.NewPCG(seed, 1))
	a, b := make([]float32, n), make([]float32, n)
	for i := range a {
		a[i], b[i] = r.Float32()*200-100, r.Float32()*200-100
	}
	var bufs [3]*Buffer
	defer func() {
		for _, buf := range bufs {
			if buf != nil {
				err = errors.Join(err, buf.Free())
			}
		}
	}()
	for i := range bufs {
		if bufs[i], err = d.Alloc(4 * n); err != nil {
			return err
		}
	}
	da, db, dc := bufs[0], bufs[1], bufs[2]
	if err := Write(da, a); err != nil {
		return err
	}
	if err := Write(db, b); err != nil {
		return err
	}
	const block = 256
	grid := uint32((n + block - 1) / block)
	if err := f.Launch(Dim3{grid, 1, 1}, Dim3{block, 1, 1}, 0, da, db, dc, uint32(n)); err != nil {
		return err
	}
	c := make([]float32, n)
	if err := Read(dc, c); err != nil {
		return err
	}
	for i := range c {
		if want := a[i] + b[i]; c[i] != want {
			return fmt.Errorf("n=%d: c[%d] = %v, want %v + %v = %v", n, i, c[i], a[i], b[i], want)
		}
	}
	return nil
}

func TestVectorAdd(t *testing.T) {
	d := openDevice(t)
	major, minor := d.ComputeCapability()
	v, err := DriverVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s, compute capability %d.%d, %d MiB, driver CUDA %d.%d",
		d.Name(), major, minor, d.TotalMemory()>>20, v/1000, v%1000/10)

	f := loadVadd(t, d)
	// 1<<20 + 3 is not a multiple of the block size, so the bounds check runs.
	for _, n := range []int{1, 255, 1<<20 + 3} {
		if err := vectorAdd(d, f, n, uint64(n)); err != nil {
			t.Fatal(err)
		}
	}
	if err := d.Synchronize(); err != nil {
		t.Fatal(err)
	}
}

// TestVectorAddConcurrent launches from many goroutines at once. They run
// on whatever threads the scheduler picks, each of which needs the context
// made current, and the driver's own threads run beside the runtime's.
func TestVectorAddConcurrent(t *testing.T) {
	d := openDevice(t)
	f := loadVadd(t, d)
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for g := range 16 {
		wg.Go(func() {
			for i := range 8 {
				if err := vectorAdd(d, f, 4096+g*8+i, uint64(g*100+i)); err != nil {
					errs <- err
					return
				}
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestErrors(t *testing.T) {
	d := openDevice(t)

	_, err := d.LoadPTX(".version 7.0\n.target sm_52\n.address_size 64\n.visible .entry bad() { not.an.instruction; }\n")
	var cerr *Error
	if !errors.As(err, &cerr) || cerr.Name != "CUDA_ERROR_INVALID_PTX" {
		t.Errorf("LoadPTX of bad PTX: err = %v, want CUDA_ERROR_INVALID_PTX", err)
	} else if !strings.Contains(err.Error(), "ptxas") {
		t.Errorf("LoadPTX error has no compile log: %v", err)
	}

	m, err := d.LoadPTX(vaddPTX)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Unload()
	if _, err := m.Function("missing"); !errors.As(err, &cerr) || cerr.Name != "CUDA_ERROR_NOT_FOUND" {
		t.Errorf("Function(missing): err = %v, want CUDA_ERROR_NOT_FOUND", err)
	}

	if _, err := d.Alloc(1 << 62); !errors.As(err, &cerr) || cerr.Name != "CUDA_ERROR_OUT_OF_MEMORY" {
		t.Errorf("Alloc(1<<62): err = %v, want CUDA_ERROR_OUT_OF_MEMORY", err)
	}

	b := alloc(t, d, 16)
	if err := Write(b, make([]float32, 5)); err == nil {
		t.Error("Write of 20 bytes to a 16-byte buffer succeeded")
	}

	if _, err := Open(1 << 20); !errors.Is(err, ErrUnavailable) {
		t.Errorf("Open(1<<20): err = %v, want ErrUnavailable", err)
	}
}
