"""Run scene multilayer-depth inference on a single full-frame RGB image.

Usage
-----

.. code-block:: bash

    python examples/infer_scene.py \
        --image examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
        --ckpt  path/to/r69e_v2_evermotion_ithappy_504.pt \
        --out   /tmp/wt_scene.rrd

Twenty hand-picked scene samples (4 featured + 16 more) live under
``examples/test_images/scene/`` -- see ``examples/test_images/README.md``.

The scene model (``r69e_v2_evermotion_ithappy_504``) was trained on
full-frame indoor renders.  By default this script:

* Treats the whole image as foreground (no center-crop, no near-white
  heuristic).
* Runs an ADE20K SegFormer to mark sky pixels as **invalid** before
  inference -- the scene model was trained on indoor renders without
  sky, so leaving outdoor sky pixels active leads to wildly far points
  along the horizon.  Pass ``--no-sky-segment`` to disable, or
  ``--sky-mask path.png`` to supply your own (white = sky).

You can still flip any of those for unusual inputs (e.g. cropping a single
object out of a scene).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wt import inference_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import parse_bg_color
from wt.data import (
    apply_sky_mask,
    load_rgba_image,
    preprocess_rgba_for_model,
    segment_sky_mask,
)
from wt.inference import _bypass_activation_checkpointing
from wt.viz import (
    init_recording,
    init_recording_layer_timeline,
    log_prediction,
    log_prediction_layer_timeline,
    save_rrd,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", required=True, type=Path, help="Path to scene image")
    p.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Path to checkpoint .pt (r69e scene model recommended)",
    )
    p.add_argument(
        "--config",
        choices=("r69e", "r75b", "r76"),
        default="r69e",
        help="Model config (default: r69e -- the scene model)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("infer_scene.rrd"), help="Output .rrd"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--alpha-erode",
        type=int,
        default=0,
        help="Erode the alpha mask by N pixels (rarely needed for scene inputs).",
    )
    p.add_argument(
        "--center-crop",
        action="store_true",
        help=(
            "Crop and centre the object based on the alpha foreground.  Off "
            "by default for scene mode (full-frame inputs)."
        ),
    )
    p.add_argument(
        "--bg-color",
        type=str,
        default="none",
        help=(
            "RGB triple (0-255) or 'none'.  Defaults to 'none' for scene "
            "mode -- the encoder should see the raw RGB.  Set to ``0,0,0`` "
            "if you are reusing the object-mode pipeline on cropped scenes."
        ),
    )
    p.add_argument(
        "--auto-alpha",
        action="store_true",
        help=(
            "Run the near-white background heuristic.  Off by default for "
            "scene mode; turn it on if your input is actually a cutout."
        ),
    )
    p.add_argument(
        "--sky-segment",
        dest="sky_segment",
        action="store_true",
        help=(
            "Run an ADE20K SegFormer and mark sky pixels as invalid before "
            "feeding the image to the scene model.  ON by default; the "
            "scene model was trained without sky and produces unstable "
            "depth on sky pixels otherwise."
        ),
    )
    p.add_argument(
        "--no-sky-segment", dest="sky_segment", action="store_false",
        help="Disable automatic sky segmentation.",
    )
    p.set_defaults(sky_segment=True)
    p.add_argument(
        "--sky-mask",
        type=Path,
        default=None,
        help=(
            "Optional path to a precomputed sky mask PNG (white = sky).  "
            "When provided, this overrides ``--sky-segment``."
        ),
    )
    p.add_argument(
        "--sky-model",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HuggingFace model id for the SegFormer ADE20K segmenter.",
    )
    p.add_argument(
        "--save-sky-mask",
        type=Path,
        default=None,
        help="If set, save the predicted/loaded sky mask to this path for inspection.",
    )
    p.add_argument(
        "--layer-timeline",
        action="store_true",
        help=(
            "Log the prediction along a ``layer`` timeline so the viewer can "
            "scrub through the layers one at a time."
        ),
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wt] config={args.config}, device={device}")

    model, cfg = build_model_and_load_ckpt(args.config, args.ckpt, device)

    rgba = load_rgba_image(args.image, auto_alpha=args.auto_alpha)
    print(f"[wt] input image: {rgba.shape}")
    bg_color = parse_bg_color(args.bg_color)

    if args.sky_mask is not None:
        sky_img = np.array(Image.open(args.sky_mask).convert("L"))
        if sky_img.shape != rgba.shape[:2]:
            raise SystemExit(
                f"sky mask shape {sky_img.shape} != image {rgba.shape[:2]}"
            )
        sky_mask = sky_img > 127
        print(f"[wt] loaded sky mask: {sky_mask.mean():.2%} of pixels")
    elif args.sky_segment:
        sky_mask = segment_sky_mask(
            rgba[:, :, :3], model_name=args.sky_model, device=device
        )
        print(f"[wt] segmented sky: {sky_mask.mean():.2%} of pixels")
    else:
        sky_mask = None

    if sky_mask is not None:
        if args.save_sky_mask is not None:
            args.save_sky_mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((sky_mask * 255).astype(np.uint8)).save(args.save_sky_mask)
            print(f"[wt] wrote sky mask preview to {args.save_sky_mask}")
        sky_bg = (0, 0, 0) if bg_color is None else bg_color
        rgba = apply_sky_mask(rgba, sky_mask, bg_color=sky_bg)

    rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        alpha_erode_px=args.alpha_erode,
        center_crop=args.center_crop,
        bg_color=bg_color,
    )

    rgb_t = rgb_t.to(device)
    mask_t = mask_t.to(device)
    intr_t = intr_t.to(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("[wt] running diffusion sampling ...")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
        xyz_pred, mask_pred, _ = inference_diffusion(
            model,
            rgb_t,
            gt_mask=mask_t,
            use_gt_mask=True,
            intrinsics=intr_t,
            invalid_fill_mode="noise",
            **cfg["inference_kwargs"],
        )

    xyz_np = xyz_pred[0].float().cpu().numpy()
    mask_np = mask_pred[0].cpu().numpy().astype(bool)

    K_solved, fov_x = solve_intrinsics_from_xyz(
        xyz_np[0], mask_np[0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    rgb_for_viz = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if args.layer_timeline:
        rec = init_recording_layer_timeline(
            application_id=f"wt.{args.config}.scene.layers"
        )
        log_prediction_layer_timeline(
            rgb_uint8=rgb_for_viz,
            xyz=xyz_np,
            mask=mask_np,
            intrinsics=K_for_viz,
            name=args.image.name,
            recording=rec,
        )
    else:
        rec = init_recording(application_id=f"wt.{args.config}.scene")
        log_prediction(
            rgb_uint8=rgb_for_viz,
            xyz=xyz_np,
            mask=mask_np,
            intrinsics=K_for_viz,
            name=args.image.name,
            recording=rec,
        )
    rrd_path = save_rrd(rec, args.out)
    print(f"[wt] wrote {rrd_path}")
    print(f"     view with: rerun {rrd_path}")


if __name__ == "__main__":
    main()
