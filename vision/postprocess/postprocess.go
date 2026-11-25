// Package postprocess converts raw object-detector outputs (class logits and
// normalized bounding boxes) into filtered Detection values.
package postprocess

import (
	"fmt"
	"math"

	"github.com/mhpenta/infer/ops"
	"github.com/mhpenta/infer/tensor"
)

// Box is a bounding box in corner coordinates (XMin, YMin, XMax, YMax),
// normalized to [0, 1] of the image dimensions unless unclipped by caller.
type Box struct {
	XMin, YMin, XMax, YMax float32
}

// Clip clamps all four coordinates of b to [0, 1] and ensures XMin <= XMax and
// YMin <= YMax.
func (b Box) Clip() Box {
	c := func(v float32) float32 {
		if v < 0 {
			return 0
		}
		if v > 1 {
			return 1
		}
		return v
	}
	out := Box{XMin: c(b.XMin), YMin: c(b.YMin), XMax: c(b.XMax), YMax: c(b.YMax)}
	if out.XMax < out.XMin {
		out.XMax = out.XMin
	}
	if out.YMax < out.YMin {
		out.YMax = out.YMin
	}
	return out
}

// IoU returns the intersection-over-union of a and b in [0, 1].
func IoU(a, b Box) float32 {
	ix0 := max(a.XMin, b.XMin)
	iy0 := max(a.YMin, b.YMin)
	ix1 := min(a.XMax, b.XMax)
	iy1 := min(a.YMax, b.YMax)
	iw := max(ix1-ix0, 0)
	ih := max(iy1-iy0, 0)
	inter := iw * ih
	areaA := max(a.XMax-a.XMin, 0) * max(a.YMax-a.YMin, 0)
	areaB := max(b.XMax-b.XMin, 0) * max(b.YMax-b.YMin, 0)
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// Detection is a single decoded object detection.
type Detection struct {
	ClassID int
	Label   string
	Score   float32
	Box     Box
}

// Options controls DETR/YOLOS postprocessing.
type Options struct {
	// ScoreThreshold keeps queries whose top non-background probability is
	// strictly greater than ScoreThreshold (matching Hugging Face's
	// post_process_object_detection).
	ScoreThreshold float32
	// Labels maps class IDs to human-readable class names (optional).
	Labels map[int]string
	// ClipBoxes clamps decoded corner boxes to [0, 1]. Leave false to match
	// Hugging Face's raw center_to_corners_format output.
	ClipBoxes bool
}

// DecodeDETR decodes DETR / YOLOS outputs: logits [queries, classes+1] (where
// the last column is the "no-object" background class) and predBoxes
// [queries, 4] in normalized (center_x, center_y, width, height) format.
// No NMS is applied: set-prediction transformer detectors predict one box per
// query.
func DecodeDETR(logits, predBoxes *tensor.Tensor, opts Options) ([]Detection, error) {
	if len(logits.Shape) != 2 || len(predBoxes.Shape) != 2 {
		return nil, fmt.Errorf("postprocess.DecodeDETR: want 2D logits and boxes, got %v and %v", logits.Shape, predBoxes.Shape)
	}
	queries, cols := logits.Rows(), logits.Cols()
	if cols < 2 {
		return nil, fmt.Errorf("postprocess.DecodeDETR: logits need at least 2 columns (classes + background), got %d", cols)
	}
	if predBoxes.Rows() != queries || predBoxes.Cols() != 4 {
		return nil, fmt.Errorf("postprocess.DecodeDETR: predBoxes %v does not match %d queries × 4", predBoxes.Shape, queries)
	}

	numClasses := cols - 1
	probs := make([]float32, cols)
	var out []Detection
	for i := range queries {
		copy(probs, logits.Row(i))
		ops.Softmax(probs)

		bestID := 0
		bestScore := probs[0]
		for c := 1; c < numClasses; c++ {
			if probs[c] > bestScore {
				bestScore = probs[c]
				bestID = c
			}
		}
		if bestScore <= opts.ScoreThreshold {
			continue
		}
		raw := predBoxes.Row(i)
		cx, cy, w, h := raw[0], raw[1], raw[2], raw[3]
		box := Box{
			XMin: cx - 0.5*w,
			YMin: cy - 0.5*h,
			XMax: cx + 0.5*w,
			YMax: cy + 0.5*h,
		}
		if opts.ClipBoxes {
			box = box.Clip()
		}
		var label string
		if opts.Labels != nil {
			label = opts.Labels[bestID]
		}
		out = append(out, Detection{
			ClassID: bestID,
			Label:   label,
			Score:   bestScore,
			Box:     box,
		})
	}
	return out, nil
}

// Sigmoid applies 1 / (1 + exp(-x)) in place.
func Sigmoid(x []float32) {
	for i, v := range x {
		x[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
	}
}
