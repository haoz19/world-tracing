"""Run multilayer-geometry inference on a single RGBA image.

Usage
-----

.. code-block:: bash

    # default: 4-seed sweep (seeds 42,43,44,45), spread along +X in one .rrd
    python examples/infer_rgba.py \
        --image examples/test_images/object/obj014_leather_briefcase.png \
        --ckpt  hf://haoz19/object-model-6layer \
        --out   /tmp/wt_demo.rrd

    # single deterministic seed (fastest path)
    python examples/infer_rgba.py \
        --image examples/test_images/object/obj014_leather_briefcase.png \
        --ckpt  hf://haoz19/object-model-6layer \
        --seed  7 \
        --out   /tmp/wt_demo.rrd

Open the resulting ``.rrd`` with ``rerun /tmp/wt_demo.rrd`` (or simply double
click in your file manager once the ``rerun-sdk`` viewer is installed).

Hand-picked object samples live under ``examples/test_images/object/``;
see ``examples/test_images/README.md`` for the full list and provenance.

The example targets the ``r75b`` config -- the object model.  Pass
``--config r69e`` or ``--config r76`` for the scene / dynamic models.
``--ckpt`` accepts a bare config name (``r75b`` / ``r69e`` / ``r76``)
which downloads the released weights from Hugging Face Hub, an explicit
``hf://owner/repo`` URI, or a local checkpoint path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wt import inference_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import add_common_args, parse_bg_color, resolve_seeds
from wt.data import load_rgba_image, preprocess_rgba_for_model
from wt.inference import _bypass_activation_checkpointing
from wt.viz import (
    init_recording,
    init_recording_layer_timeline,
    log_multiseed_prediction,
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

    seeds = resolve_seeds(args)
    if args.layer_timeline and len(seeds) > 1:
        raise SystemExit(
            "--layer-timeline is incompatible with the default multi-seed "
            "sweep.  Pass --seed N (or --num-seeds 1) for a single-seed "
            "layer-timeline visualisation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[wt] config={args.config}, device={device}, "
        f"seeds={seeds} ({len(seeds)} sample{'s' if len(seeds) > 1 else ''})"
    )

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

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    seeds_xyz: list[np.ndarray] = []
    seeds_mask: list[np.ndarray] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        print(f"[wt] running diffusion (seed={seed}) ...")
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
        seeds_xyz.append(xyz_pred[0].float().cpu().numpy())  # [L, H, W, 3]
        seeds_mask.append(mask_pred[0].cpu().numpy().astype(bool))  # [L, H, W]

    K_solved, fov_x = solve_intrinsics_from_xyz(
        seeds_xyz[0][0], seeds_mask[0][0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from seed-0/layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    rgb_for_viz = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if len(seeds) == 1:
        if args.layer_timeline:
            rec = init_recording_layer_timeline(
                application_id=f"wt.{args.config}.layers"
            )
            log_prediction_layer_timeline(
                rgb_uint8=rgb_for_viz,
                xyz=seeds_xyz[0],
                mask=seeds_mask[0],
                intrinsics=K_for_viz,
                name=args.image.name,
                recording=rec,
            )
        else:
            rec = init_recording(application_id=f"wt.{args.config}")
            log_prediction(
                rgb_uint8=rgb_for_viz,
                xyz=seeds_xyz[0],
                mask=seeds_mask[0],
                intrinsics=K_for_viz,
                name=args.image.name,
                recording=rec,
            )
    else:
        rec = init_recording(application_id=f"wt.{args.config}.multiseed")
        log_multiseed_prediction(
            rgb_uint8=rgb_for_viz,
            seeds_xyz=seeds_xyz,
            seeds_mask=seeds_mask,
            seed_values=seeds,
            intrinsics=K_for_viz,
            name=args.image.name,
            recording=rec,
        )

    rrd_path = save_rrd(rec, args.out)
    print(f"[wt] wrote {rrd_path}")
    print(f"     view with: rerun {rrd_path}")


if __name__ == "__main__":
    main()
