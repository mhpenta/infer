//go:build unix

package metal

import (
	"os"
	"syscall"
	"testing"
)

// TestWrap has the GPU write into an anonymous mapping, the kind of memory a
// memory-mapped weights file is, without copying it.
func TestWrap(t *testing.T) {
	d := open(t)
	scale := pipelines(t, d, "scale")[0]
	page := os.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mem)
	if _, err := d.Wrap(mem[4:]); err == nil {
		t.Error("Wrap accepted memory that is not page-aligned")
	}
	if _, err := d.Wrap(mem[:page+4]); err == nil {
		t.Error("Wrap accepted a length that is not a whole number of pages")
	}
	buf, err := d.Wrap(mem)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Release()
	xv := Slice[float32](buf)
	if &buf.Bytes()[0] != &mem[0] || len(xv) != len(mem)/4 {
		t.Fatal("wrapped buffer does not share the mapping")
	}
	for i := range xv {
		xv[i] = 3
	}
	cb := d.NewCommandBuffer()
	cb.SetPipeline(scale)
	cb.SetBuffer(0, buf, 0)
	SetValue(cb, 1, struct {
		s float32
		n uint32
	}{2, uint32(len(xv))})
	cb.DispatchThreads(Size{len(xv), 1, 1}, Size{256, 1, 1})
	if err := cb.Wait(); err != nil {
		t.Fatal(err)
	}
	for i, v := range xv {
		if v != 6 {
			t.Fatalf("x[%d] = %v, want 6", i, v)
		}
	}
}
