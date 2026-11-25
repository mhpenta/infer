// Command detect runs YOLOS object detection on one or more JPEG/PNG images
// and prints JSON detections per image.
//
//	go run ./cmd/detect [-threshold 0.5] [-device auto] <model_dir> <image> [more images...]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"time"

	"github.com/mhpenta/infer/model/yolos"
	"github.com/mhpenta/infer/vision/postprocess"
)

type output struct {
	Image      string                  `json:"image"`
	Device     string                  `json:"device"`
	Width      int                     `json:"width"`
	Height     int                     `json:"height"`
	ElapsedMS  float64                 `json:"elapsed_ms"`
	Detections []postprocess.Detection `json:"detections"`
}

func main() {
	threshold := flag.Float64("threshold", 0.5, "minimum detection score")
	device := flag.String("device", "", "device: cpu, metal, or auto")
	flag.Parse()
	if flag.NArg() < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s [-threshold 0.5] [-device auto] <model_dir> <image> [more images...]\n", os.Args[0])
		os.Exit(2)
	}
	if err := run(flag.Arg(0), flag.Args()[1:], float32(*threshold), *device); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(dir string, paths []string, threshold float32, device string) error {
	model, err := yolos.LoadWith(dir, yolos.Options{Device: device})
	if err != nil {
		return err
	}
	defer model.Close()

	enc := json.NewEncoder(os.Stdout)
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		img, _, err := image.Decode(f)
		f.Close()
		if err != nil {
			return fmt.Errorf("decode %s: %w", p, err)
		}
		t0 := time.Now()
		dets, err := model.Detect(img, threshold)
		elapsed := float64(time.Since(t0).Microseconds()) / 1000.0
		if err != nil {
			return fmt.Errorf("detect %s: %w", p, err)
		}
		b := img.Bounds()
		if err := enc.Encode(output{
			Image:      p,
			Device:     model.Device(),
			Width:      b.Dx(),
			Height:     b.Dy(),
			ElapsedMS:  elapsed,
			Detections: dets,
		}); err != nil {
			return err
		}
	}
	return nil
}
