//go:build !unix

package safetensors

import "os"

// mapFile reads the whole file on platforms without mmap support here.
func mapFile(path string) ([]byte, func() error, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}
