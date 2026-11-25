package postprocess

import (
	"math"
	"testing"

	"github.com/mhpenta/infer/tensor"
)

func TestDecodeDETR(t *testing.T) {
	// 3 queries, 2 object classes + 1 background class (index 2).
	logits := tensor.FromData([]float32{
		-1, 5, 0, // class 1 wins strongly
		0, 0, 10, // background wins (excluded)
		2, 1, 2, // class 0 ≈ 0.422 (below 0.5, above 0.4)
	}, 3, 3)
	boxes := tensor.FromData([]float32{
		0.5, 0.5, 0.4, 0.6, // -> [0.3, 0.2, 0.7, 0.8]
		0.2, 0.2, 0.1, 0.1,
		0.05, 0.95, 0.2, 0.2, // -> [-0.05, 0.85, 0.15, 1.05]
	}, 3, 4)

	labels := map[int]string{0: "cat", 1: "person"}
	dets, err := DecodeDETR(logits, boxes, Options{
		ScoreThreshold: 0.5,
		Labels:         labels,
		ClipBoxes:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	d := dets[0]
	if d.ClassID != 1 || d.Label != "person" || d.Score < 0.98 {
		t.Fatalf("unexpected detection: %+v", d)
	}
	if math.Abs(float64(d.Box.XMin-0.3)) > 1e-6 || math.Abs(float64(d.Box.YMax-0.8)) > 1e-6 {
		t.Fatalf("unexpected box: %+v", d.Box)
	}

	// Lower threshold picks up query 2; ClipBoxes clamps to [0, 1].
	dets, err = DecodeDETR(logits, boxes, Options{
		ScoreThreshold: 0.4,
		Labels:         labels,
		ClipBoxes:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(dets) != 2 {
		t.Fatalf("got %d detections, want 2", len(dets))
	}
	if dets[1].Box.XMin != 0 || dets[1].Box.YMax != 1 {
		t.Errorf("unclipped box: %+v", dets[1].Box)
	}
}

func TestIoU(t *testing.T) {
	a := Box{0, 0, 0.5, 0.5}
	if got := IoU(a, a); math.Abs(float64(got-1)) > 1e-6 {
		t.Errorf("identical IoU = %v, want 1", got)
	}
	b := Box{0.5, 0.5, 1, 1}
	if got := IoU(a, b); got != 0 {
		t.Errorf("disjoint IoU = %v, want 0", got)
	}
	c := Box{0.25, 0, 0.75, 0.5} // half overlap -> inter=0.125, union=0.375 -> 1/3
	if got := IoU(a, c); math.Abs(float64(got-1.0/3.0)) > 1e-6 {
		t.Errorf("half-overlap IoU = %v, want 1/3", got)
	}
}
