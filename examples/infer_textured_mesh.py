"""Run end-to-end ``image -> textured GLB`` inference.

This example chains the released multilayer-geometry model with the public
`TRELLIS.2 <https://github.com/microsoft/TRELLIS.2>`_ pipeline:

1. Run ``inference_diffusion`` to predict per-layer XYZ + masks.
2. Densify between adjacent layers (``expand_cloud_ray_xyz``).
3. Canonicalise into ``[-0.5, 0.5]^3`` (``CanonicalTransform``).
4. Voxelise + close + fill (``v4_ray_fill``).
5. Inject those voxel coords into TRELLIS.2 stage 2 + 3 and save a GLB.

Usage
-----

.. code-block:: bash

    # 1. Install TRELLIS.2 separately (one-time, see its README):
    #    git clone https://github.com/microsoft/TRELLIS.2
    #    cd TRELLIS.2
    #    bash setup.sh --new-env --basic --flash-attn --o-voxel \\
    #                  --nvdiffrast --cumesh --flexgemm
    #
    # 2. Activate the trellis2 env, then run a 4-seed sweep (default):
    python examples/infer_textured_mesh.py \\
        --image  examples/test_images/objects/case_new.png \\
        --ckpt   hf://haoz19/object-model-6layer \\
        --config r75b \\
        --out    /tmp/wt_textured.glb \\
        --trellis2-path /path/to/TRELLIS.2

The default 4-seed sweep writes ``/tmp/wt_textured_seed{42,43,44,45}.glb``
so you can compare meshes and keep the best one.  Pass ``--seed N`` (or
``--num-seeds 1``) to run a single deterministic seed and write to the
plain ``--out`` path instead.

You may also pass ``--rrd path.rrd`` to additionally dump the predicted
multilayer point cloud for sanity checking inside Rerun (single-seed
inference only; the multi-seed sweep emits ``path_seed{N}.rrd`` for each
seed when this flag is set).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from wt import inference_diffusion
from wt.checkpoint import build_model_and_load_ckpt
from wt.cli import parse_bg_color, resolve_seeds
from wt.data import load_rgba_image, preprocess_rgba_for_model
from wt.inference import _bypass_activation_checkpointing
from wt.textured_mesh import (
    aggregate_camera_cloud,
    compute_canonical_transform,
    inject_coords_into_trellis2,
    load_trellis2_pipeline,
    save_mesh_glb,
    v4_ray_fill,
)


def _seed_path(out: Path, seed: int, total_seeds: int) -> Path:
    """For a multi-seed sweep return ``<stem>_seed{N}<suffix>``; else ``out``."""
    if total_seeds == 1:
        return out
    return out.with_name(f"{out.stem}_seed{seed}{out.suffix}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to an RGBA / RGB object image (BiRefNet auto-matting is "
        "applied for RGB inputs when --auto-alpha is set).",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help=(
            "Checkpoint -- one of: hf://owner/repo, a bare config name "
            "(`r75b`, `r69e`, `r76`) for the default HF repo, or a local "
            "path to a .pt / .safetensors file."
        ),
    )
    p.add_argument(
        "--config",
        choices=("r75b", "r69e", "r76"),
        default="r75b",
        help="Model config (default: r75b -- the object model).",
    )
    p.add_argument(
        "--out", type=Path, default=Path("infer_textured.glb"), help="Output .glb"
    )
    p.add_argument(
        "--rrd",
        type=Path,
        default=None,
        help="Optional .rrd path to also dump the multilayer point cloud.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run a single deterministic seed.  When neither ``--seed`` nor "
            "``--num-seeds`` is set, the script runs the default 4-seed "
            "sweep and writes ``<out_stem>_seed{N}.glb`` for each seed."
        ),
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of independent TRELLIS.2 / diffusion seeds (default: 4 "
            "when neither ``--seed`` nor ``--num-seeds`` is set).  Each "
            "seed produces its own GLB."
        ),
    )
    p.add_argument(
        "--bg-color", type=str, default="0,0,0", help="RGB or 'none' for background fill"
    )
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument(
        "--no-auto-alpha", dest="auto_alpha", action="store_false"
    )
    p.add_argument("--alpha-erode", type=int, default=2)
    p.add_argument("--center-crop", action="store_true", default=True)

    p.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="1024_cascade",
        help="TRELLIS.2 pipeline variant (default: 1024_cascade -- best quality / time trade-off).",
    )
    p.add_argument(
        "--ss-res",
        type=int,
        default=64,
        help="Sparse-structure voxel grid resolution (32 for 512, 64 for 1024+).",
    )
    p.add_argument(
        "--ray-steps",
        type=int,
        default=4,
        help="Interior samples per layer-pair during ray densification.",
    )
    p.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Optional cap on the number of voxel coords fed into TRELLIS.2.",
    )
    p.add_argument(
        "--trellis2-model",
        type=str,
        default="microsoft/TRELLIS.2-4B",
        help="TRELLIS.2 HF repo id (or local path).",
    )
    p.add_argument(
        "--trellis2-path",
        type=Path,
        default=None,
        help="Path to a local TRELLIS.2 checkout (prepended to sys.path).",
    )
    p.add_argument(
        "--attn-backend",
        choices=("flash_attn", "xformers"),
        default=None,
        help="Override TRELLIS.2's sparse-attention backend.",
    )
    p.add_argument(
        "--texture-size", type=int, default=4096, help="Baked texture resolution."
    )
    p.add_argument(
        "--decimation-target",
        type=int,
        default=1_000_000,
        help="Target face count after decimation (default: 1M).",
    )
    args = p.parse_args()

    seeds = resolve_seeds(args)
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

    print(f"[wt] loading TRELLIS.2 pipeline ({args.trellis2_model}) ...")
    pipeline = load_trellis2_pipeline(
        model_id=args.trellis2_model,
        device=device,
        trellis2_path=args.trellis2_path,
        attn_backend=args.attn_backend,
    )

    from PIL import Image

    pil_rgba = Image.fromarray(rgba, mode="RGBA")

    for seed in seeds:
        glb_path = _seed_path(args.out, seed, len(seeds))
        rrd_path = (
            _seed_path(args.rrd, seed, len(seeds))
            if args.rrd is not None
            else None
        )
        print(f"\n[wt] === seed {seed} ===")

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        print("[wt] running diffusion sampling ...")
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
        print(
            f"[wt] predicted XYZ {xyz_np.shape}, valid frac={mask_np.mean():.2%}"
        )

        if rrd_path is not None:
            from wt.viz import init_recording, log_prediction, save_rrd
            from wt import solve_intrinsics_from_xyz

            K_solved, _ = solve_intrinsics_from_xyz(
                xyz_np[0], mask_np[0], image_size=cfg["image_size"]
            )
            K_for_viz = (
                K_solved if K_solved is not None else intr_t[0].cpu().numpy()
            )
            rgb_for_viz = (
                rgb_t[0].permute(1, 2, 0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            rec = init_recording(
                application_id=f"wt.{args.config}.textured_mesh"
            )
            log_prediction(
                rgb_uint8=rgb_for_viz,
                xyz=xyz_np,
                mask=mask_np,
                intrinsics=K_for_viz,
                name=args.image.name,
                recording=rec,
            )
            save_rrd(rec, rrd_path)
            print(f"[wt] wrote {rrd_path}")

        cloud_cam = aggregate_camera_cloud(xyz_np, mask_np)
        if cloud_cam.size == 0:
            print(
                f"[wt] seed {seed}: empty point cloud -- skipping voxelisation."
            )
            continue
        tf = compute_canonical_transform(cloud_cam, half_target=0.45)
        coords_xyz, n_vox = v4_ray_fill(
            xyz_np,
            mask_np,
            tf.apply,
            res=args.ss_res,
            ray_steps=args.ray_steps,
            max_voxels=args.max_voxels,
            seed=seed,
        )
        print(
            f"[wt] v4_ray_fill: {n_vox:,} voxels (sent to TRELLIS.2: "
            f"{coords_xyz.shape[0]:,})"
        )
        if coords_xyz.shape[0] == 0:
            print(f"[wt] seed {seed}: v4_ray_fill produced 0 voxels -- skipping.")
            continue

        print(
            f"[wt] running TRELLIS.2 ({args.pipeline_type}, seed={seed}) ..."
        )
        meshes = inject_coords_into_trellis2(
            pipeline,
            pil_rgba,
            coords_xyz,
            seed=seed,
            pipeline_type=args.pipeline_type,
            preprocess_image=False,  # rgba is already alpha-matted
        )
        mesh = meshes[0]

        out = save_mesh_glb(
            glb_path,
            mesh,
            texture_size=args.texture_size,
            decimation_target=args.decimation_target,
        )
        print(f"[wt] wrote {out}")


if __name__ == "__main__":
    main()
