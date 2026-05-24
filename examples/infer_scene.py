"""Run scene multilayer-geometry inference on a single full-frame RGB image.

Usage
-----

.. code-block:: bash

    # default: 4-seed sweep (seeds 42,43,44,45), spread along +X in one .rrd
    python examples/infer_scene.py \
        --image examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
        --ckpt  hf://haoz19/scene-model-6layer \
        --out   /tmp/wt_scene.rrd

    # single deterministic seed (fastest path)
    python examples/infer_scene.py \
        --image examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
        --ckpt  hf://haoz19/scene-model-6layer \
        --seed  7 \
        --out   /tmp/wt_scene.rrd

Hand-picked scene samples live under ``examples/test_images/scene/`` --
see ``examples/test_images/README.md``.

The scene model (``r69e``) was trained on full-frame indoor renders.
By default this script treats the whole image as foreground (no
center-crop, no auto-matting); the released scene model was trained on
indoor renders without sky, so for outdoor images with large sky regions
you should pre-mask the sky externally (any matting / segmentation tool
of your choice).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wt import inference_diffusion, solve_intrinsics_from_xyz
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import parse_bg_color, resolve_seeds
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
    p.add_argument("--image", required=True, type=Path, help="Path to scene image")
    p.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help=(
            "Checkpoint -- a local .pt path, an HF shorthand "
            "``hf://owner/repo``, or a bare config name "
            "(``r69e`` recommended for scene mode)."
        ),
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
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run a single deterministic seed.  When neither ``--seed`` nor "
            "``--num-seeds`` is set, the script runs 4 seeds (``42, 43, "
            "44, 45``) so you can compare diffusion samples side-by-side "
            "and pick the best."
        ),
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of independent diffusion seeds (default: 4 when "
            "neither ``--seed`` nor ``--num-seeds`` is set).  Pass "
            "``--num-seeds 1`` for the fastest single-sample mode."
        ),
    )
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
            "Run BiRefNet-based foreground matting (or the near-white "
            "heuristic if applicable).  Off by default for scene mode; "
            "turn it on if your input is actually a cutout."
        ),
    )
    p.add_argument(
        "--layer-timeline",
        action="store_true",
        help=(
            "Log the prediction along a ``layer`` timeline so the viewer can "
            "scrub through the layers one at a time.  Requires single-seed "
            "mode (combine with ``--seed N`` or ``--num-seeds 1``)."
        ),
    )
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

    rgba = load_rgba_image(args.image, auto_alpha=args.auto_alpha)
    print(f"[wt] input image: {rgba.shape}")
    bg_color = parse_bg_color(args.bg_color)

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
        seeds_xyz.append(xyz_pred[0].float().cpu().numpy())
        seeds_mask.append(mask_pred[0].cpu().numpy().astype(bool))

    K_solved, fov_x = solve_intrinsics_from_xyz(
        seeds_xyz[0][0], seeds_mask[0][0], image_size=cfg["image_size"]
    )
    print(f"[wt] solved K from seed-0/layer-0 XYZ; fov_x ≈ {fov_x:.1f}°")
    K_for_viz = K_solved if K_solved is not None else intr_t[0].cpu().numpy()

    rgb_for_viz = (rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    if len(seeds) == 1:
        if args.layer_timeline:
            rec = init_recording_layer_timeline(
                application_id=f"wt.{args.config}.scene.layers"
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
            rec = init_recording(application_id=f"wt.{args.config}.scene")
            log_prediction(
                rgb_uint8=rgb_for_viz,
                xyz=seeds_xyz[0],
                mask=seeds_mask[0],
                intrinsics=K_for_viz,
                name=args.image.name,
                recording=rec,
            )
    else:
        rec = init_recording(application_id=f"wt.{args.config}.scene.multiseed")
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
