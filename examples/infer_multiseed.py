"""Run multilayer-depth inference with N different seeds on a single image.

The diffusion sampler is stochastic, so different random seeds produce
visually different point clouds in occluded layers.  This example runs
``--num_seeds`` independent denoising trajectories on the same image and
logs all of them to one Rerun recording, spread along the ``+X`` axis so
they can be compared at a glance.

Usage
-----

.. code-block:: bash

    python examples/infer_multiseed.py \
        --image     examples/test_images/object/obj063_trex_dinosaur.png \
        --ckpt      path/to/r75b_spikeskip_hardaug.pt \
        --num_seeds 4 \
        --out       /tmp/wt_multiseed.rrd
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
from wt.viz import init_recording, log_multiseed_prediction, save_rrd


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--image", required=True, type=Path, help="Path to RGB/RGBA image"
    )
    p.add_argument(
        "--num_seeds",
        type=int,
        default=4,
        help="Number of independent diffusion seeds (default: 4).",
    )
    add_common_args(p, default_out="infer_multiseed.rrd")
    args = p.parse_args()

    if args.num_seeds < 1:
        raise SystemExit("--num_seeds must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[wt] config={args.config}, device={device}, num_seeds={args.num_seeds}")

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
    seed_values: list[int] = []

    for i in range(args.num_seeds):
        seed = args.seed + i
        seed_values.append(seed)
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
        seeds_xyz.append(xyz_pred[0].float().cpu().numpy())
        seeds_mask.append(mask_pred[0].cpu().numpy().astype(bool))

    K_solved, fov_x = solve_intrinsics_from_xyz(
        seeds_xyz[0][0], seeds_mask[0][0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from seed-0/layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    rgb_for_viz = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    rec = init_recording(application_id=f"wt.{args.config}.multiseed")
    log_multiseed_prediction(
        rgb_uint8=rgb_for_viz,
        seeds_xyz=seeds_xyz,
        seeds_mask=seeds_mask,
        seed_values=seed_values,
        intrinsics=K_for_viz,
        name=args.image.name,
        recording=rec,
    )
    rrd_path = save_rrd(rec, args.out)
    print(f"[wt] wrote {rrd_path}")
    print(f"     view with: rerun {rrd_path}")


if __name__ == "__main__":
    main()
