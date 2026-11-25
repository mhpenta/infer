package yolos

import (
	"errors"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/mhpenta/infer/accel"
	"github.com/mhpenta/infer/internal/golden"
	"github.com/mhpenta/infer/metal"
	"github.com/mhpenta/infer/safetensors"
	"github.com/mhpenta/infer/tensor"
	"github.com/mhpenta/infer/vision/postprocess"
	"github.com/mhpenta/infer/vision/preprocess"
)

// relTol is the largest difference from PyTorch allowed in a hidden state,
// logits or predicted boxes, relative to that tensor's largest magnitude (or 1
// when smaller), matching model/llama's parity check.
const relTol = 2e-5

type goldenDetection struct {
	ClassID int        `json:"class_id"`
	Label   string     `json:"label"`
	Score   float32    `json:"score"`
	Box     [4]float32 `json:"box"`
}

type goldenCase struct {
	Name         string            `json:"name"`
	Image        string            `json:"image"`
	OrigWidth    int               `json:"orig_width"`
	OrigHeight   int               `json:"orig_height"`
	ProcWidth    int               `json:"proc_width"`
	ProcHeight   int               `json:"proc_height"`
	DoResize     bool              `json:"do_resize"`
	ShortestEdge int               `json:"shortest_edge"`
	LongestEdge  int               `json:"longest_edge"`
	HiddenLayers []int             `json:"hidden_layers"`
	Threshold    float32           `json:"threshold"`
	Detections   []goldenDetection `json:"detections"`
}

func testDevices(t testing.TB) []string {
	_, err := accel.OpenMetal()
	if errors.Is(err, metal.ErrUnavailable) && os.Getenv("INFER_REQUIRE_METAL") != "1" {
		return []string{accel.CPU}
	}
	if err != nil {
		t.Fatal(err)
	}
	return []string{accel.CPU, accel.Metal}
}

func loadOn(t testing.TB, dir, device string) *Model {
	t.Helper()
	model, err := LoadWith(dir, Options{Device: device})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(model.Device(), device) {
		t.Fatalf("loaded on %s, want %s", model.Device(), device)
	}
	t.Cleanup(func() { model.Close() })
	return model
}

// TestYolosTinyMatchesPython feeds PyTorch's exact pixel_values tensors into
// HiddenStatesPixels on CPU and Metal and checks patch embeddings, interpolated
// position embeddings, every golden encoder hidden state, final LayerNorm,
// class logits, predicted boxes, and decoded detections.
func TestYolosTinyMatchesPython(t *testing.T) {
	for _, device := range testDevices(t) {
		t.Run(device, func(t *testing.T) {
			checkYolos(t, device, "INFER_YOLOS_TINY_DIR", "yolos-tiny", "yolos_tiny")
		})
	}
}

// TestYolosSmall300MatchesPython verifies hustvl/yolos-small-300, including
// mid_position_embeddings across all 12 encoder layers, on CPU and Metal.
func TestYolosSmall300MatchesPython(t *testing.T) {
	for _, device := range testDevices(t) {
		t.Run(device, func(t *testing.T) {
			checkYolos(t, device, "INFER_YOLOS_SMALL_300_DIR", "yolos-small-300", "yolos_small_300")
		})
	}
}

func checkYolos(t *testing.T, device, envVar, modelName, goldenPrefix string) {
	dir := golden.ModelDir(t, envVar, modelName)
	model := loadOn(t, dir, device)

	var cases []goldenCase
	golden.JSON(t, goldenPrefix+"_model.json", &cases)
	want, err := safetensors.Open(golden.Path(t, goldenPrefix+"_model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()

	for i, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			pixels, err := want.Tensor(fmt.Sprintf("case%d.pixel_values", i))
			if err != nil {
				t.Fatal(err)
			}
			got, err := model.HiddenStatesPixels(pixels)
			if err != nil {
				t.Fatal(err)
			}

			checkTensor(t, "patch_embed", got.PatchEmbed, want, fmt.Sprintf("case%d.patch_embed", i))
			checkTensor(t, "pos_embed", got.PosEmbed, want, fmt.Sprintf("case%d.pos_embed", i))

			if len(got.HiddenStates) != model.Config.NumHiddenLayers+1 {
				t.Fatalf("got %d hidden states, want %d", len(got.HiddenStates), model.Config.NumHiddenLayers+1)
			}
			var worstLayer float64
			for _, j := range c.HiddenLayers {
				rel := checkTensor(t, fmt.Sprintf("hidden.%d", j), got.HiddenStates[j], want, fmt.Sprintf("case%d.hidden.%d", i, j))
				worstLayer = max(worstLayer, rel)
			}
			finalRel := checkTensor(t, "last_hidden_state", got.LastHiddenState, want, fmt.Sprintf("case%d.last_hidden_state", i))
			worstLayer = max(worstLayer, finalRel)

			logitsRel := checkTensor(t, "logits", got.Logits, want, fmt.Sprintf("case%d.logits", i))
			boxesRel := checkTensor(t, "pred_boxes", got.PredBoxes, want, fmt.Sprintf("case%d.pred_boxes", i))

			// Verify postprocessing matches PyTorch's post_process_object_detection.
			dets, err := postprocess.DecodeDETR(got.Logits, got.PredBoxes, postprocess.Options{
				ScoreThreshold: c.Threshold,
				Labels:         model.Config.Labels(),
				ClipBoxes:      false,
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(dets) != len(c.Detections) {
				t.Fatalf("got %d detections, want %d", len(dets), len(c.Detections))
			}
			for k, d := range dets {
				ref := c.Detections[k]
				if d.ClassID != ref.ClassID || d.Label != ref.Label {
					t.Errorf("det[%d]: class (%d, %q), want (%d, %q)", k, d.ClassID, d.Label, ref.ClassID, ref.Label)
				}
				if diff := math.Abs(float64(d.Score) - float64(ref.Score)); diff > 1e-4 {
					t.Errorf("det[%d]: score %g, want %g (diff %g)", k, d.Score, ref.Score, diff)
				}
				refBox := postprocess.Box{XMin: ref.Box[0], YMin: ref.Box[1], XMax: ref.Box[2], YMax: ref.Box[3]}
				if iou := postprocess.IoU(d.Box, refBox); iou < 0.999 {
					t.Errorf("det[%d]: box %+v vs %+v, IoU %.6f", k, d.Box, refBox, iou)
				}
			}
			t.Logf("%s (%dx%d): layers worst rel %.2g; logits rel %.2g; boxes max |diff| %.2g; %d/%d detections exact",
				c.Name, c.ProcWidth, c.ProcHeight, worstLayer, logitsRel, boxesRel, len(dets), len(c.Detections))
		})
	}
}

// TestYolosTinyEndToEndFromPNG runs the full Go pipeline (PNG decode ->
// preprocess.Process -> model.ForwardPixels -> postprocess.DecodeDETR) on
// both CPU and Metal and verifies the non-resized case matches PyTorch tightly
// and CPU vs Metal agree on every resized case.
func TestYolosTinyEndToEndFromPNG(t *testing.T) {
	devices := testDevices(t)
	dir := golden.ModelDir(t, "INFER_YOLOS_TINY_DIR", "yolos-tiny")
	models := make(map[string]*Model, len(devices))
	for _, dev := range devices {
		models[dev] = loadOn(t, dir, dev)
	}

	var cases []goldenCase
	golden.JSON(t, "yolos_tiny_model.json", &cases)
	goldenDir := filepath.Dir(golden.Path(t, "yolos_tiny_model.json"))

	for _, c := range cases {
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

			preCfg := preprocess.DefaultYOLOSConfig()
			preCfg.DoResize = c.DoResize
			if c.DoResize {
				preCfg.Size = preprocess.Size{ShortestEdge: c.ShortestEdge, LongestEdge: c.LongestEdge}
			}

			pre, err := preprocess.Process(img, preCfg)
			if err != nil {
				t.Fatal(err)
			}

			var cpuOut *Output
			for _, dev := range devices {
				out, err := models[dev].ForwardPixels(pre.Pixels)
				if err != nil {
					t.Fatal(err)
				}
				if dev == accel.CPU {
					cpuOut = out
				} else {
					if rel := relMaxDiff(out.Logits.Data, cpuOut.Logits.Data); rel > 2*relTol {
						t.Errorf("Metal vs CPU logits rel diff %g > %g", rel, 2*relTol)
					}
					if rel := relMaxDiff(out.PredBoxes.Data, cpuOut.PredBoxes.Data); rel > 2*relTol {
						t.Errorf("Metal vs CPU pred_boxes rel diff %g > %g", rel, 2*relTol)
					}
				}
				// Verify DetectWith runs cleanly end-to-end.
				dets, err := models[dev].DetectWith(img, preCfg, postprocess.Options{
					ScoreThreshold: c.Threshold,
					ClipBoxes:      false,
				})
				if err != nil {
					t.Fatal(err)
				}
				if !c.DoResize && len(dets) != len(c.Detections) {
					t.Errorf("%s: got %d detections, want %d", dev, len(dets), len(c.Detections))
				}
			}
		})
	}
}

func TestRejectsBadInput(t *testing.T) {
	model, err := Load(golden.ModelDir(t, "INFER_YOLOS_TINY_DIR", "yolos-tiny"))
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if _, err := model.ForwardPixels(nil); err == nil {
		t.Error("nil tensor succeeded, want error")
	}
	if _, err := model.ForwardPixels(tensor.New(3, 8, 8)); err == nil {
		t.Error("sub-patch tensor succeeded, want error")
	}
	if _, err := model.ForwardPixels(tensor.New(1, 32, 32)); err == nil {
		t.Error("1-channel tensor succeeded, want error")
	}
}

func TestGPUWorkspacesDoNotAccumulate(t *testing.T) {
	if len(testDevices(t)) < 2 {
		t.Skip("needs Metal")
	}
	model := loadOn(t, golden.ModelDir(t, "INFER_YOLOS_TINY_DIR", "yolos-tiny"), accel.Metal)
	free := func() []int {
		model.gpu.mu.Lock()
		defer model.gpu.mu.Unlock()
		var rows []int
		for _, w := range model.gpu.workspace {
			rows = append(rows, w.rows)
		}
		return rows
	}
	// Run growing, then shrinking image sizes.
	for _, hw := range [][2]int{{64, 64}, {128, 128}, {192, 256}, {96, 96}, {192, 256}, {64, 64}} {
		if _, err := model.ForwardPixels(tensor.New(3, hw[0], hw[1])); err != nil {
			t.Fatal(err)
		}
	}
	wantRows := 1 + (192/16)*(256/16) + model.Config.NumDetectionTokens
	if got := free(); !slices.Equal(got, []int{wantRows}) {
		t.Errorf("free workspaces hold %v rows, want [%d]", got, wantRows)
	}
}

func BenchmarkYolosTinyForward(b *testing.B) {
	for _, device := range testDevices(b) {
		b.Run(device, func(b *testing.B) {
			model := loadOn(b, golden.ModelDir(b, "INFER_YOLOS_TINY_DIR", "yolos-tiny"), device)
			for _, tc := range []struct {
				name string
				h, w int
			}{
				{"160x128", 128, 160},
				{"360x640", 360, 640},
				{"512x896", 512, 896},
			} {
				pixels := tensor.New(3, tc.h, tc.w)
				// Warmup (fills pos-embed cache and GPU workspace).
				if _, err := model.ForwardPixels(pixels); err != nil {
					b.Fatal(err)
				}
				b.Run(tc.name, func(b *testing.B) {
					for b.Loop() {
						if _, err := model.ForwardPixels(pixels); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}

func checkTensor(t *testing.T, name string, got *tensor.Tensor, want *safetensors.File, key string) float64 {
	t.Helper()
	ref, err := want.Tensor(key)
	if err != nil {
		t.Fatalf("%s: load golden %s: %v", name, key, err)
	}
	if !slices.Equal(got.Shape, ref.Shape) {
		t.Fatalf("%s: shape %v, want %v", name, got.Shape, ref.Shape)
	}
	rel := relMaxDiff(got.Data, ref.Data)
	if math.IsNaN(rel) || rel > relTol {
		t.Fatalf("%s: max |diff| / scale = %.3g > %.3g (max |diff| %.3g, max |want| %.3g)",
			name, rel, relTol, maxDiff(got.Data, ref.Data), maxAbs(ref.Data))
	}
	return rel
}

func relMaxDiff(got, want []float32) float64 {
	scale := max(maxAbs(want), 1)
	return maxDiff(got, want) / scale
}

func maxAbs(x []float32) float64 {
	var m float64
	for _, v := range x {
		m = max(m, math.Abs(float64(v)))
	}
	return m
}

func maxDiff(got, want []float32) float64 {
	var m float64
	for k := range got {
		d := math.Abs(float64(got[k]) - float64(want[k]))
		if math.IsNaN(d) {
			return d
		}
		m = max(m, d)
	}
	return m
}
