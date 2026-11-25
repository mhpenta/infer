"""
Dump reference outputs from PyTorch for hustvl/yolos-tiny so the Go
implementation (vision/preprocess + model/yolos + postprocessing) can be
checked layer-by-layer against transformers on CPU and Metal.

    .venv/bin/python scripts/dump_golden_yolos.py \
        testdata/models/yolos-tiny testdata/golden/yolos_tiny

Writes:
  <out_prefix>_images/<name>.png   deterministic synthetic test images
  <out_prefix>_model.json          case metadata and postprocessed detections
  <out_prefix>_model.safetensors   per case i:
                                     case{i}.pixel_values       [3, H, W]
                                     case{i}.patch_embed        [patches, 192]
                                     case{i}.pos_embed          [seq, 192]
                                     case{i}.hidden.{j}         [seq, 192]
                                     case{i}.last_hidden_state  [seq, 192]
                                     case{i}.logits             [100, 92]
                                     case{i}.pred_boxes         [100, 4]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from transformers import AutoImageProcessor, YolosForObjectDetection


def make_blank(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), (128, 128, 128))


def make_geometric(w: int, h: int) -> Image.Image:
    # Deterministic non-uniform scene with gradients and sharp edges to
    # exercise resize antialiasing, patch boundaries, and bicubic pos-embed.
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    r = np.clip(0.6 * x + 0.2 * y, 0, 255).astype(np.uint8)
    g = np.clip(0.3 * (255 - x) + 0.5 * y, 0, 255).astype(np.uint8)
    b = np.clip(180 - 0.4 * y + 0.2 * x, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([w // 8, h // 5, w // 3, 4 * h // 5], fill=(220, 40, 30), outline=(255, 255, 255), width=3)
    draw.ellipse([w // 2, h // 4, 7 * w // 8, 3 * h // 4], fill=(30, 160, 230), outline=(10, 10, 10), width=4)
    draw.polygon([(w // 3, h // 6), (w // 2, h // 2), (w // 4, 5 * h // 6)], fill=(240, 210, 40))
    return img


def make_camera_frame(w: int, h: int) -> Image.Image:
    # 16:9 camera-like scene (640x360) with horizon, room blocks, and contrast.
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[: h // 2, :, :] = (210, 220, 235)
    arr[h // 2 :, :, :] = (95, 85, 75)
    # Subtle sinusoidal texture across patches
    yy, xx = np.mgrid[0:h, 0:w]
    wave = (15 * np.sin(xx / 11.0) * np.cos(yy / 9.0)).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + wave[..., None], 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 80, 200, 320], fill=(45, 70, 120), outline=(240, 240, 240), width=2)
    draw.rectangle([380, 110, 560, 340], fill=(160, 90, 50), outline=(20, 20, 20), width=2)
    draw.ellipse([240, 140, 340, 260], fill=(200, 170, 140))
    return img


CASES = [
    # (name, image_fn, orig_w, orig_h, do_resize, shortest_edge, longest_edge, all_layers)
    ("blank_160x128", make_blank, 160, 128, False, 0, 0, True),
    ("nonsquare_240x160", make_geometric, 240, 160, True, 160, 320, True),
    ("portrait_180x320", make_geometric, 180, 320, True, 192, 384, True),
    ("camera_640x360", make_camera_frame, 640, 360, True, 512, 1333, False),
]


def main():
    args = [a for a in sys.argv[1:] if a != "--compact"]
    compact = "--compact" in sys.argv[1:]
    if len(args) != 2:
        sys.exit(f"usage: python {sys.argv[0]} [--compact] <model_dir> <out_prefix>")
    model_dir, out = args[0], args[1]

    proc = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = YolosForObjectDetection.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    img_dir = f"{out}_images"
    os.makedirs(img_dir, exist_ok=True)
    img_dir_rel = os.path.basename(img_dir)

    tensors = {}
    cases_meta = []

    selected = CASES[:2] if compact else CASES
    for i, (name, fn, w, h, do_resize, short_e, long_e, all_layers) in enumerate(selected):
        img = fn(w, h)
        img_path = os.path.join(img_dir, f"{name}.png")
        img.save(img_path)

        proc_kwargs = {"images": img, "return_tensors": "pt", "do_resize": do_resize}
        if do_resize:
            proc_kwargs["size"] = {"shortest_edge": short_e, "longest_edge": long_e}
        enc = proc(**proc_kwargs)
        pixel_values = enc["pixel_values"]  # [1, 3, ph, pw]
        ph, pw = int(pixel_values.shape[-2]), int(pixel_values.shape[-1])

        captured = {}

        def hook_patch(_m, _inp, out_t):
            captured["patch_embed"] = out_t.detach()

        def hook_pos(_m, _inp, out_t):
            captured["pos_embed"] = out_t.detach()

        h1 = model.vit.embeddings.patch_embeddings.register_forward_hook(hook_patch)
        h2 = model.vit.embeddings.interpolation.register_forward_hook(hook_pos)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, output_hidden_states=True)
        h1.remove()
        h2.remove()

        tensors[f"case{i}.pixel_values"] = pixel_values[0].contiguous().to(torch.float32)
        tensors[f"case{i}.patch_embed"] = captured["patch_embed"][0].contiguous().to(torch.float32)
        tensors[f"case{i}.pos_embed"] = captured["pos_embed"][0].contiguous().to(torch.float32)

        layer_indices = list(range(len(outputs.hidden_states))) if all_layers else [0, 6, len(outputs.hidden_states) - 1]
        for j in layer_indices:
            tensors[f"case{i}.hidden.{j}"] = outputs.hidden_states[j][0].contiguous().to(torch.float32)

        # Final normalized hidden state before detection heads:
        # note outputs.hidden_states[-1] in YolosModel is BEFORE vit.layernorm!
        with torch.no_grad():
            last_normed = model.vit.layernorm(outputs.hidden_states[-1])
        tensors[f"case{i}.last_hidden_state"] = last_normed[0].contiguous().to(torch.float32)
        tensors[f"case{i}.logits"] = outputs.logits[0].contiguous().to(torch.float32)
        tensors[f"case{i}.pred_boxes"] = outputs.pred_boxes[0].contiguous().to(torch.float32)

        # Normalized [0, 1] postprocessed detections at threshold=0.05 (so we get
        # several boxes even on synthetic scenes to verify postprocessing parity).
        threshold = 0.05
        res = proc.post_process_object_detection(outputs, threshold=threshold, target_sizes=None)[0]
        dets = []
        for score, label, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
            dets.append({
                "class_id": int(label),
                "label": model.config.id2label[int(label)],
                "score": float(score),
                "box": [float(v) for v in box],
            })

        cases_meta.append({
            "name": name,
            "image": f"{img_dir_rel}/{name}.png",
            "orig_width": w,
            "orig_height": h,
            "proc_width": pw,
            "proc_height": ph,
            "do_resize": do_resize,
            "shortest_edge": short_e,
            "longest_edge": long_e,
            "hidden_layers": layer_indices,
            "threshold": threshold,
            "detections": dets,
        })

    save_file(tensors, f"{out}_model.safetensors")
    with open(f"{out}_model.json", "w") as f:
        json.dump(cases_meta, f, indent=2)
    print(f"wrote {len(cases_meta)} cases to {out}_model.* ({os.path.getsize(out + '_model.safetensors') / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
