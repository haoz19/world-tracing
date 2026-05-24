"""Run dynamic-clip multilayer-geometry inference on a folder of RGBA frames.

Usage
-----

.. code-block:: bash

    # default: 4-seed sweep (seeds 42,43,44,45), spread along +X in one .rrd
    python examples/infer_video.py \
        --image_dir examples/test_images/dynamic/davis__camel/ \
        --ckpt     hf://haoz19/dynamic-model-16frame \
        --config   r76 \
        --out      /tmp/wt_video.rrd

    # single deterministic seed (fastest path)
    python examples/infer_video.py \
        --image_dir examples/test_images/dynamic/davis__camel/ \
        --ckpt     hf://haoz19/dynamic-model-16frame \
        --config   r76 \
        --seed     7 \
        --out      /tmp/wt_video.rrd

Hand-picked 16-frame dynamic clips live under
``examples/test_images/dynamic/`` -- see ``examples/test_images/README.md``.

``frame_indices`` is optional; without it all frames in the directory are
loaded in sorted order.

The output ``.rrd`` uses the ``frame`` timeline (one entry per loaded
frame), so scrubbing through it animates the predicted point cloud over
time.  Open with ``rerun /tmp/wt_video.rrd``.

This entry point only makes sense with the ``r76`` config (which has the
temporal-attention blocks); ``r75b`` / ``r69e`` are still accepted but
behave identically to running ``infer_rgba.py`` once per frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wt import inference_video_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import add_common_args, parse_bg_color, resolve_seeds
from wt.data import load_video_clip, preprocess_clip_for_model
from wt.inference import _bypass_activation_checkpointing
from wt.viz import (
    init_recording_video,
    log_video_clip_prediction,
    log_video_multiseed_prediction,
    save_rrd,
)


def _parse_frame_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    if raw.strip() == "":
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--image_dir",
        required=True,
        type=Path,
        help="Directory of RGBA/RGB frames (one image per frame).",
    )
    p.add_argument(
        "--frame_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated 0-based indices into the sorted file list to "
            "use as the clip (e.g. '0,2,4,6').  Default: every frame."
        ),
    )
    add_common_args(p, default_out="infer_video.rrd")
    args = p.parse_args()
    if args.config != "r76":
        print(
            f"[wt] WARNING: --config={args.config} does not have temporal "
            "blocks; r76 is the released video model."
        )

    seeds = resolve_seeds(args)
    if args.layer_timeline:
        print(
            "[wt] WARNING: --layer-timeline is a no-op for video inference; "
            "the video recording always uses the per-frame timeline."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[wt] config={args.config}, device={device}, "
        f"seeds={seeds} ({len(seeds)} sample{'s' if len(seeds) > 1 else ''})"
    )

    model, cfg = build_model_and_load_ckpt(args.config, args.ckpt, device)

    frame_indices = _parse_frame_indices(args.frame_indices)
    frame_names, rgba_list = load_video_clip(
        args.image_dir,
        frame_indices=frame_indices,
        auto_alpha=not args.no_auto_alpha,
    )
    print(f"[wt] loaded {len(rgba_list)} frame(s) from {args.image_dir}")

    bg_color = parse_bg_color(args.bg_color)
    rgb_clip, mask_clip, intr_t, rgb_resized = preprocess_clip_for_model(
        rgba_list,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        alpha_erode_px=args.alpha_erode,
        center_crop=not args.no_center_crop,
        bg_color=bg_color,
    )

    rgb_clip = rgb_clip.to(device)
    mask_clip = mask_clip.to(device)
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
        print(
            f"[wt] running clip diffusion sampling on T={len(rgba_list)} "
            f"frames (seed={seed}) ..."
        )
        with torch.no_grad(), autocast_ctx, _bypass_activation_checkpointing(model):
            xyz_pred, mask_pred, _ = inference_video_diffusion(
                model,
                rgb_clip,
                gt_mask_clip=mask_clip,
                use_gt_mask=True,
                intrinsics=intr_t,
                invalid_fill_mode="noise",
                **cfg["inference_kwargs"],
            )
        seeds_xyz.append(xyz_pred[0].float().cpu().numpy())  # [T, L, H, W, 3]
        seeds_mask.append(mask_pred[0].cpu().numpy().astype(bool))  # [T, L, H, W]

    K_solved, fov_x = solve_intrinsics_from_xyz(
        seeds_xyz[0][0, 0], seeds_mask[0][0, 0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from seed-0/frame-0/layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    if len(seeds) == 1:
        rec = init_recording_video(application_id=f"wt.{args.config}.clip")
        log_video_clip_prediction(
            rgb_clip=rgb_resized,
            xyz_clip=seeds_xyz[0],
            mask_clip=seeds_mask[0],
            intrinsics=K_for_viz,
            frame_names=frame_names,
            name=args.image_dir.name,
            recording=rec,
        )
    else:
        rec = init_recording_video(
            application_id=f"wt.{args.config}.clip.multiseed"
        )
        log_video_multiseed_prediction(
            rgb_clip=rgb_resized,
            seeds_xyz=seeds_xyz,
            seeds_mask=seeds_mask,
            seed_values=seeds,
            intrinsics=K_for_viz,
            frame_names=frame_names,
            name=args.image_dir.name,
            recording=rec,
        )

    rrd_path = save_rrd(rec, args.out)
    print(f"[wt] wrote {rrd_path}")
    print(f"     view with: rerun {rrd_path}")


if __name__ == "__main__":
    main()
