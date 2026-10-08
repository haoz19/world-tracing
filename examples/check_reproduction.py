"""Check that your install reproduces our reference outputs of the dynamic model.

Downloads a few DAVIS clips -- 16 RGBA frames each (every 2nd frame, DAVIS
ground-truth masks as alpha), i.e. exactly the inputs behind our DAVIS
results -- together with the depth maps we obtained for them with the released
``r76`` checkpoint, runs the same clips through your install and reports how
far your depth maps are from ours.

Usage
-----

.. code-block:: bash

    # all reference clips x 4 seeds (~35 s per clip and seed on an H100)
    python examples/check_reproduction.py

    # quick check: one seed per clip
    python examples/check_reproduction.py --seeds 42

How to read the result: the same code and weights reproduce the reference up
to GPU floating-point noise (well below ``PASS_REL_DIFF``).  Anything that
changes the sampling -- outdated code or weights, a different noise prior,
different preprocessing -- gives a different sample, which differs from the
reference about as much as another seed does (the per-clip "seed gap" printed
next to each result).

Your predictions are written to ``--out`` so they can be sent back to us.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from wt import inference_video_diffusion
from wt.checkpoint import build_model_and_load_ckpt
from wt.data import load_video_clip, preprocess_clip_for_model
from wt.inference import _bypass_activation_checkpointing

REF_REPO = "haoz19/dynamic-model-16frame"
REF_SUBDIR = "reproduce/davis"
BG_COLOR = (128, 128, 128)
PASS_REL_DIFF = 0.01


def fetch_reference(ref: str | None) -> Path:
    if ref is not None:
        return Path(ref)
    from huggingface_hub import snapshot_download

    root = snapshot_download(REF_REPO, allow_patterns=[f"{REF_SUBDIR}/*"])
    return Path(root) / REF_SUBDIR


@torch.no_grad()
def predict(model, cfg: dict, frames_dir: Path, seeds: list[int], device: torch.device):
    """Run one clip like ``examples/infer_video.py`` with its default flags.

    Returns the model-resolution RGB the network saw ``[T, H, W, 3]`` and, per
    seed, depth ``[S, T, L, H, W]`` (float16, 0 outside the mask) and mask.
    """
    _, rgba = load_video_clip(frames_dir)
    rgb, mask, intr, rgb_resized = preprocess_clip_for_model(
        rgba,
        image_size=cfg["image_size"],
        num_layers=cfg["model_kwargs"]["num_layers"],
        bg_color=BG_COLOR,
    )
    depths, masks = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            _bypass_activation_checkpointing(model),
        ):
            xyz, m, _ = inference_video_diffusion(
                model,
                rgb.to(device),
                gt_mask_clip=mask.to(device),
                use_gt_mask=True,
                intrinsics=intr.to(device),
                invalid_fill_mode="noise",
                **cfg["inference_kwargs"],
            )
        m = m[0].cpu().numpy().astype(bool)
        z = xyz[0, ..., 2].float().cpu().numpy()
        depths.append(np.where(m, z, 0.0).astype(np.float16))
        masks.append(m)
    return np.stack(rgb_resized), np.stack(depths), np.stack(masks)


def rel_depth_diff(za, ma, zb, mb, layer: int | None = None) -> float:
    """Mean ``|za - zb| / zb`` over pixels valid in both masks."""
    if layer is not None:
        za, ma, zb, mb = za[:, layer], ma[:, layer], zb[:, layer], mb[:, layer]
    both = ma & mb
    a = za[both].astype(np.float32)
    b = zb[both].astype(np.float32)
    return float((np.abs(a - b) / np.maximum(np.abs(b), 1e-6)).mean())


def seed_gap(depth: np.ndarray, mask: np.ndarray) -> float:
    """Mean L0 relative difference between pairs of reference seeds."""
    gaps = [
        rel_depth_diff(depth[i], mask[i], depth[j], mask[j], layer=0)
        for i in range(len(depth))
        for j in range(i + 1, len(depth))
    ]
    return float(np.mean(gaps)) if gaps else float("nan")


def environment() -> dict:
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    try:
        env["wt_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        env["wt_commit"] = "unknown (not a git checkout)"
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", default="r76", help="Checkpoint (default: released r76 from HF).")
    p.add_argument("--ref", default=None, help="Local reference dir (default: download from HF).")
    p.add_argument("--clips", default=None, help="Comma-separated clip names (default: all).")
    p.add_argument(
        "--seeds", default=None, help="Comma-separated seeds (default: all reference seeds)."
    )
    p.add_argument("--out", type=Path, default=Path("check_reproduction_out"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "A CUDA GPU is required (the reference was computed with bf16 autocast on GPU)."
        )
    device = torch.device("cuda")

    ref_root = fetch_reference(args.ref)
    manifest = json.loads((ref_root / "manifest.json").read_text())
    clips = args.clips.split(",") if args.clips else manifest["clips"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else manifest["seeds"]
    print(f"[wt] reference: {ref_root} (made with {manifest['made_with']})")

    env = environment()
    print(f"[wt] environment: {env}")
    model, cfg = build_model_and_load_ckpt("r76", args.ckpt, device)
    print(f"[wt] inference_kwargs: {cfg['inference_kwargs']}")
    if cfg["inference_kwargs"].get("noise_time_corr", 0.0) != manifest["noise_time_corr"]:
        print(
            "[wt] WARNING: this code samples with noise_time_corr="
            f"{cfg['inference_kwargs'].get('noise_time_corr', 0.0)}, the reference used "
            f"{manifest['noise_time_corr']}.  Update the code (git pull)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    results, ok = [], True
    for clip in clips:
        ref = np.load(ref_root / clip / "reference.npz")
        rgb, depth, mask = predict(model, cfg, ref_root / clip / "frames", seeds, device)
        np.savez_compressed(
            args.out / f"{clip}.npz", seeds=np.array(seeds), rgb=rgb, depth=depth, mask=mask
        )

        rgb_diff = int(np.abs(rgb.astype(np.int16) - ref["rgb"].astype(np.int16)).max())
        gap = seed_gap(ref["depth"], ref["mask"])
        if rgb_diff > 1:
            print(
                f"[wt] WARNING: {clip}: model input differs from ours (max |RGB diff| {rgb_diff}); check preprocessing."
            )
        for i, seed in enumerate(seeds):
            j = list(ref["seeds"]).index(seed)
            rd, rm = ref["depth"][j], ref["mask"][j]
            iou = float((mask[i] & rm).sum() / max((mask[i] | rm).sum(), 1))
            l0 = rel_depth_diff(depth[i], mask[i], rd, rm, layer=0)
            all_layers = rel_depth_diff(depth[i], mask[i], rd, rm)
            passed = l0 < PASS_REL_DIFF and all_layers < PASS_REL_DIFF and iou > 0.999
            ok &= passed
            results.append(
                dict(
                    clip=clip,
                    seed=seed,
                    input_rgb_max_diff=rgb_diff,
                    mask_iou=iou,
                    l0_rel_diff=l0,
                    all_layers_rel_diff=all_layers,
                    seed_gap_l0=gap,
                    passed=passed,
                )
            )
            print(
                f"[wt] {clip:15s} seed {seed}: L0 depth diff {100 * l0:6.3f}%  all layers "
                f"{100 * all_layers:6.3f}%  mask IoU {iou:.4f}  (another seed: {100 * gap:.1f}%)  "
                f"{'PASS' if passed else 'FAIL'}"
            )

    (args.out / "summary.json").write_text(
        json.dumps({"environment": env, "results": results}, indent=1)
    )
    print(f"[wt] predictions + summary written to {args.out}/")
    if ok:
        print("[wt] PASS: your install reproduces the reference outputs.")
    else:
        print(
            "[wt] FAIL: outputs differ from the reference.  Please send us "
            f"{args.out}/summary.json and the full log."
        )


if __name__ == "__main__":
    main()
