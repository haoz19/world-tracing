"""Run multilayer-depth inference on a single RGBA image.

Usage
-----

.. code-block:: bash

    python examples/infer_rgba.py \
        --image examples/test_images/object/obj014_leather_briefcase.png \
        --ckpt  path/to/r75b_spikeskip_hardaug.pt \
        --out   /tmp/wt_demo.rrd

Open the resulting ``.rrd`` with ``rerun /tmp/wt_demo.rrd`` (or simply double
click in your file manager once the ``rerun-sdk`` viewer is installed).

Thirty hand-picked object samples (12 featured + 18 more) live under
``examples/test_images/object/``; see
``examples/test_images/README.md`` for the full list and provenance.

The example targets the ``r75b_spikeskip_hardaug`` config -- the object
model.  Pass ``--config r69e`` or ``--config r76`` for the scene / dynamic
models.  Checkpoints are TODO: download links will be added once the public
checkpoint URLs are finalised.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wt import inference_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import add_common_args, parse_bg_color
from wt.data import load_rgba_image, preprocess_rgba_for_model
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
    p.add_argument(
        "--image", required=True, type=Path, help="Path to RGB/RGBA image"
    )
    add_common_args(p, default_out="infer_rgba.rrd")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wt] config={args.config}, device={device}")

    model, cfg = build_model_and_load_ckpt(args.config, args.ckpt, device)

    rgba = load_rgba_image(args.image, auto_alpha=not args.no_auto_alpha)
    print(f"[wt] input image: {rgba.shape}")
    bg_color = parse_bg_color(args.bg_color)

    rgb_t, mask_t, intr_t = preprocess_rgba_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        alpha_erode_px=args.alpha_erode,
        center_crop=not args.no_center_crop,
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

    xyz_np = xyz_pred[0].float().cpu().numpy()  # [L, H, W, 3]
    mask_np = mask_pred[0].cpu().numpy().astype(bool)  # [L, H, W]

    K_solved, fov_x = solve_intrinsics_from_xyz(
        xyz_np[0], mask_np[0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    rgb_for_viz = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if args.layer_timeline:
        rec = init_recording_layer_timeline(application_id=f"wt.{args.config}.layers")
        log_prediction_layer_timeline(
            rgb_uint8=rgb_for_viz,
            xyz=xyz_np,
            mask=mask_np,
            intrinsics=K_for_viz,
            name=args.image.name,
            recording=rec,
        )
    else:
        rec = init_recording(application_id=f"wt.{args.config}")
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
