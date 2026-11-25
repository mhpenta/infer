package preprocess

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/safetensors"
)

func TestTargetSize(t *testing.T) {
	tests := []struct {
		name               string
		origH, origW       int
		size               Size
		mod                int
		wantH, wantW       int
	}{
		{"camera_16_9", 360, 640, Size{ShortestEdge: 512, LongestEdge: 1333}, 16, 512, 896},
		{"square_upscale", 256, 256, Size{ShortestEdge: 512, LongestEdge: 1333}, 16, 512, 512},
		{"portrait", 640, 360, Size{ShortestEdge: 512, LongestEdge: 1333}, 16, 896, 512},
		{"very_wide_clamped", 200, 1000, Size{ShortestEdge: 512, LongestEdge: 1333}, 16, 256, 1328},
		{"explicit_hw", 360, 640, Size{Height: 360, Width: 640}, 16, 352, 640},
		{"no_mod", 360, 640, Size{ShortestEdge: 512, LongestEdge: 1333}, 0, 512, 910},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h, w := TargetSize(tc.origH, tc.origW, tc.size, tc.mod)
			if h != tc.wantH || w != tc.wantW {
				t.Errorf("TargetSize(%d, %d) = (%d, %d), want (%d, %d)", tc.origH, tc.origW, h, w, tc.wantH, tc.wantW)
			}
		})
	}
}

func TestExtractPatches(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 32, 16))
	for y := range 16 {
		for x := range 32 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	cfg := Config{DoResize: false, DoRescale: false, DoNormalize: false}
	res, err := Process(img, cfg)
	if err != nil {
		t.Fatal(err)
	}
	patches, gh, gw, err := ExtractPatches(res.Pixels, 16)
	if err != nil {
		t.Fatal(err)
	}
	if gh != 1 || gw != 2 || patches.Rows() != 2 || patches.Cols() != 3*16*16 {
		t.Fatalf("got grid %dx%d, shape %v", gh, gw, patches.Shape)
	}
	// Check patch 1 (x=16..31, y=0..15), pixel (px=3, py=2) -> x=19, y=2.
	p1 := patches.Row(1)
	if got := p1[0*256+2*16+3]; got != 19 {
		t.Errorf("R = %v, want 19", got)
	}
	if got := p1[1*256+2*16+3]; got != 2 {
		t.Errorf("G = %v, want 2", got)
	}
	if got := p1[2*256+2*16+3]; got != 21 {
		t.Errorf("B = %v, want 21", got)
	}
}

func TestMatchesPythonPreprocess(t *testing.T) {
	type goldenCase struct {
		Name         string `json:"name"`
		Image        string `json:"image"`
		ProcWidth    int    `json:"proc_width"`
		ProcHeight   int    `json:"proc_height"`
		DoResize     bool   `json:"do_resize"`
		ShortestEdge int    `json:"shortest_edge"`
		LongestEdge  int    `json:"longest_edge"`
	}
	var cases []goldenCase
	golden.JSON(t, "yolos_tiny_model.json", &cases)
	want, err := safetensors.Open(golden.Path(t, "yolos_tiny_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()

	goldenDir := filepath.Dir(golden.Path(t, "yolos_tiny_model.json"))
	for i, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			f, err := os.Open(filepath.Join(goldenDir, c.Image))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			img, err := png.Decode(f)
			if err != nil {
				t.Fatal(err)
			}

			cfg := DefaultYOLOSConfig()
			cfg.DoResize = c.DoResize
			if c.DoResize {
				cfg.Size = Size{ShortestEdge: c.ShortestEdge, LongestEdge: c.LongestEdge}
			}
			got, err := Process(img, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if got.Width != c.ProcWidth || got.Height != c.ProcHeight {
				t.Fatalf("size = %dx%d, want %dx%d", got.Width, got.Height, c.ProcWidth, c.ProcHeight)
			}

			ref, err := want.Tensor(caseKey(i, "pixel_values"))
			if err != nil {
				t.Fatal(err)
			}
			var maxAbs, sumAbs float64
			for k := range got.Pixels.Data {
				d := math.Abs(float64(got.Pixels.Data[k]) - float64(ref.Data[k]))
				if d > maxAbs {
					maxAbs = d
				}
				sumAbs += d
			}
			meanAbs := sumAbs / float64(len(got.Pixels.Data))
			t.Logf("%s (%dx%d): max |diff| %.4g, mean |diff| %.4g", c.Name, got.Width, got.Height, maxAbs, meanAbs)

			// Without resize, normalization must match PyTorch to float32 precision.
			// With uint8 bilinear AA resize, 1/255/0.224 ≈ 0.0175 is one uint8 rounding step.
			limit := 0.02
			if !c.DoResize {
				limit = 1e-6
			}
			if maxAbs > limit {
				t.Errorf("max |diff| = %g, want <= %g", maxAbs, limit)
			}
		})
	}
}

func caseKey(i int, suffix string) string {
	return "case" + string(rune('0'+i)) + "." + suffix
}
