//go:build unix

package safetensors

import (
	"os"
	"syscall"
)

// mapFile maps path read-only and returns the bytes plus a function to unmap.
func mapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close() // the mapping outlives the descriptor
	info, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if info.Size() == 0 {
		return nil, func() error { return nil }, nil
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return syscall.Munmap(data) }, nil
}
