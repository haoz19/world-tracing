"""3D-FRONT held-out scene benchmark (Table 2 of the World Tracing paper).

Evaluates a scene model -- by default the released ``r69e`` 504x504 scene
model -- on the 50 held-out 3D-FRONT frames of Table 2 and reports the mean
(and std) over 8 diffusion seeds.  The ground truth (input RGB, the first 6
depth-peeled layers as camera-space XYZ, per-layer validity masks, intrinsics)
is downloaded from ``haoz19/scene-model-6layer`` (``eval/3dfront_test50``).

Usage
-----

.. code-block:: bash

    # released r69e, 8 seeds (~45 s per frame for all 8 seeds on an H100)
    python benchmarks/3dfront/eval_3dfront.py

    # quick check: one seed
    python benchmarks/3dfront/eval_3dfront.py --num-seeds 1

    # score your own method's predictions with the same protocol
    python benchmarks/3dfront/eval_3dfront.py --pred-dir my_preds/

Protocol
--------

* Samples: 10 3D-FRONT rooms held out from training (20 rendered frames
  each); 50 frames drawn from them, listed in ``samples_3dfront_test50.json``.
  Frames are 512x512 renders resized to 504x504, field of view 70-108 deg,
  centred principal point.
* GT: the first 6 depth-peeled surfaces per pixel, divided by the per-frame
  median depth, so depth errors and point distances are relative to the
  median scene depth (not metres).
* Model input: the RGB frame; the L0 valid-depth mask is broadcast to all
  6 layers as ``gt_mask``; 20 Euler steps; seed ``42 + i + 1000 * k`` for
  frame ``i`` and seed index ``k``.  Only the predicted z is scored; it is
  unprojected with the GT intrinsics.
* L0 depth (MAE / RMSE / AbsRel): per-frame least-squares scale + shift of
  the predicted L0 depth onto GT on valid pixels; per-frame metrics are
  averaged with pixel-count weights.
* L0 CD: the same L0 fit, unprojected with the GT intrinsics.
* All-layer CD: one scale + shift fitted jointly on all layers; predictions
  are kept only where the corresponding GT layer is valid.
* Chamfer: CD-L1 = mean of the two directed mean NN distances, CD-L2 the
  same with squared distances, F@0.05 the F-score at 0.05; each cloud is
  randomly subsampled to <= 50k points (``RandomState(12345)``); averaged
  over frames.

``--pred-dir`` format: one ``<stem>.npz`` per frame (stems listed in
``samples_3dfront_test50.json``, e.g. ``00_21a0379d_MasterBedroom-142318_f03``)
holding ``depth`` -- ``[L, 504, 504]`` or ``[504, 504]`` camera-space z in any
scale -- and optionally ``mask`` (same shape, bool; default: finite depth).
Layer ``l`` is compared to GT layer ``l``; single-layer predictions only get
the L0 metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DATA_REPO = "haoz19/scene-model-6layer"
DATA_SUBDIR = "eval/3dfront_test50"
SIZE = 504
NUM_LAYERS = 6
MAX_POINTS = 50000
KEYS = [
    "MAE",
    "RMSE",
    "AbsRel",
    "L0_CD_L1",
    "L0_CD_L2",
    "L0_F@0.05",
    "All_CD_L1",
    "All_CD_L2",
    "All_F@0.05",
]
PAPER_WT_S = [0.0102, 0.0215, 0.0114, 0.0093, 0.000204, 0.9867, 0.0216, 0.001255, 0.8951]


def fetch_data(data: str | None) -> Path:
    if data is not None:
        return Path(data)
    from huggingface_hub import snapshot_download

    root = snapshot_download(DATA_REPO, allow_patterns=[f"{DATA_SUBDIR}/*"])
    return Path(root) / DATA_SUBDIR


def load_samples(root: Path) -> list[dict]:
    index = json.loads((root / "samples_3dfront_test50.json").read_text())
    samples = []
    for entry in index:
        with np.load(root / entry["file"]) as z:
            samples.append({"stem": Path(entry["file"]).stem, **{k: z[k] for k in z.files}})
    return samples


# ---------------------------------------------------------------- scoring


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() < 10:
        return pred.astype(np.float32)
    A = np.stack([pred[mask].astype(np.float64), np.ones(int(mask.sum()))], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, gt[mask].astype(np.float64), rcond=None)
    return (pred.astype(np.float64) * s + b).astype(np.float32)


def unproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    v, u = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij")
    xyz = np.stack(
        [(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], axis=-1
    )
    xyz[~mask] = 0
    return xyz.astype(np.float32)


def flatten(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = xyz[mask]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > MAX_POINTS:
        pts = pts[np.random.RandomState(12345).choice(len(pts), MAX_POINTS, replace=False)]
    return pts


def chamfer(
    pred_xyz: np.ndarray, pred_mask: np.ndarray, gt_xyz: np.ndarray, gt_mask: np.ndarray
) -> dict:
    from scipy.spatial import cKDTree

    pred, gt = flatten(pred_xyz, pred_mask), flatten(gt_xyz, gt_mask)
    if len(pred) == 0 or len(gt) == 0:
        return {}
    d_p2g, _ = cKDTree(gt).query(pred, k=1)
    d_g2p, _ = cKDTree(pred).query(gt, k=1)
    p, r = float(np.mean(d_p2g < 0.05)), float(np.mean(d_g2p < 0.05))
    return {
        "cd_l1": float(0.5 * (d_p2g.mean() + d_g2p.mean())),
        "cd_l2": float(0.5 * ((d_p2g**2).mean() + (d_g2p**2).mean())),
        "f05": 2 * p * r / (p + r) if p + r > 0 else 0.0,
    }


def score(sample: dict, depth: np.ndarray, mask: np.ndarray) -> dict:
    """Score one prediction; ``depth``/``mask`` are ``[L, H, W]``."""
    gt_xyz, gt_mask, K = sample["gt_xyz"], sample["gt_mask"], sample["intrinsics"]
    gt0, m0 = gt_xyz[0, :, :, 2], gt_mask[0]

    valid = m0 & np.isfinite(gt0) & np.isfinite(depth[0]) & mask[0]
    out = {"count": int(valid.sum())}
    if out["count"] >= 10:
        aligned = fit_scale_shift(depth[0], gt0, valid)
        valid &= np.isfinite(aligned)
        diff = np.abs(aligned[valid].astype(np.float64) - gt0[valid])
        out.update(
            count=int(valid.sum()),
            mae=diff.mean(),
            rmse=np.sqrt((diff**2).mean()),
            abs_rel=(diff / np.maximum(gt0[valid], 1e-8)).mean(),
        )

    m = mask[0] & m0
    aligned = fit_scale_shift(depth[0], gt0, m)
    out["cd_l0"] = chamfer(
        unproject(aligned, m & np.isfinite(aligned), K)[None], m[None], gt_xyz[:1], m0[None]
    )

    L = min(depth.shape[0], NUM_LAYERS)
    out["cd_all"] = {}
    if L > 1:
        joint = mask[:L] & gt_mask[:L] & np.isfinite(depth[:L]) & np.isfinite(gt_xyz[:L, ..., 2])
        if joint.sum() >= 10:
            A = np.stack([depth[:L][joint].astype(np.float64), np.ones(int(joint.sum()))], axis=1)
            (s, b), *_ = np.linalg.lstsq(
                A, gt_xyz[:L, ..., 2][joint].astype(np.float64), rcond=None
            )
            al = (depth[:L].astype(np.float64) * s + b).astype(np.float32)
            pred_xyz = np.stack(
                [unproject(al[l], joint[l] & np.isfinite(al[l]), K) for l in range(L)]
            )
            out["cd_all"] = chamfer(pred_xyz, joint, gt_xyz[:L], gt_mask[:L])
    return out


def aggregate(rows: list[dict]) -> dict:
    d = [r for r in rows if r.get("count", 0) > 0 and "mae" in r]
    w = np.array([r["count"] for r in d], dtype=np.float64)

    def pix(k):
        return float(np.sum([r[k] for r in d] * w) / w.sum())

    def cd(scope, k):
        v = [r[scope][k] for r in rows if r[scope] and np.isfinite(r[scope][k])]
        return float(np.mean(v)) if v else float("nan")

    vals = [
        pix("mae"),
        pix("rmse"),
        pix("abs_rel"),
        cd("cd_l0", "cd_l1"),
        cd("cd_l0", "cd_l2"),
        cd("cd_l0", "f05"),
        cd("cd_all", "cd_l1"),
        cd("cd_all", "cd_l2"),
        cd("cd_all", "f05"),
    ]
    return dict(zip(KEYS, vals))


def fmt_row(name: str, vals) -> str:
    cells = [
        "-" if v is None or not np.isfinite(v) else (f"{v:.6f}" if "CD_L2" in k else f"{v:.4f}")
        for k, v in zip(KEYS, vals)
    ]
    return f"| {name} | " + " | ".join(cells) + " |"


# ---------------------------------------------------------------- our model


def predict_ours(model, cfg: dict, sample: dict, seed: int, device) -> np.ndarray:
    import torch

    from wt import inference_diffusion
    from wt.inference import _bypass_activation_checkpointing

    L = cfg["model_kwargs"]["num_layers"]
    rgb = torch.from_numpy(sample["rgb"]).unsqueeze(0).to(device)
    gt_mask = (
        torch.from_numpy(sample["gt_mask"][0])
        .to(device)
        .view(1, 1, SIZE, SIZE)
        .expand(1, L, SIZE, SIZE)
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        _bypass_activation_checkpointing(model),
    ):
        xyz, _, _ = inference_diffusion(
            model,
            rgb,
            gt_mask=gt_mask,
            use_gt_mask=True,
            invalid_fill_mode="noise",
            **cfg["inference_kwargs"],
        )
    return xyz[0, ..., 2].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt", type=str, default="r69e", help="local .pt, hf://repo[/file.pt] or a config name"
    )
    ap.add_argument(
        "--config",
        default="r69e",
        help="model config in wt/checkpoint.py (must be a 504x504 model)",
    )
    ap.add_argument("--num-seeds", type=int, default=8)
    ap.add_argument(
        "--pred-dir", default=None, help="score precomputed predictions instead of running a model"
    )
    ap.add_argument(
        "--data", default=None, help="local copy of eval/3dfront_test50 (default: download)"
    )
    ap.add_argument("--out", default="3dfront_eval_results.json")
    args = ap.parse_args()

    samples = load_samples(fetch_data(args.data))
    print(f"[3dfront] {len(samples)} frames")

    if args.pred_dir is not None:
        rows = []
        for s in samples:
            with np.load(Path(args.pred_dir) / f"{s['stem']}.npz") as z:
                depth = np.asarray(z["depth"], dtype=np.float32)
                depth = depth[None] if depth.ndim == 2 else depth
                mask = (
                    np.asarray(z["mask"], dtype=bool) if "mask" in z.files else np.isfinite(depth)
                )
                mask = mask[None] if mask.ndim == 2 else mask
            rows.append(score(s, depth, mask & np.isfinite(depth)))
        res = aggregate(rows)
        print(
            "\n".join(
                [
                    "| Method | " + " | ".join(KEYS) + " |",
                    "|---|" + "---:|" * len(KEYS),
                    fmt_row("paper WT-S", PAPER_WT_S),
                    fmt_row(args.pred_dir, [res[k] for k in KEYS]),
                ]
            )
        )
        Path(args.out).write_text(
            json.dumps({"pred_dir": args.pred_dir, "metrics": res, "per_frame": rows}, indent=1)
        )
        return

    import torch

    from wt.checkpoint import build_model_and_load_ckpt

    device = torch.device("cuda")
    model, cfg = build_model_and_load_ckpt(args.config, args.ckpt, device)
    if cfg["image_size"] != SIZE:
        raise SystemExit(
            f"--config {args.config} runs at {cfg['image_size']}px; this benchmark is defined at {SIZE}px"
        )

    per_seed: list[list[dict]] = [[] for _ in range(args.num_seeds)]
    t0 = time.time()
    for i, s in enumerate(samples):
        for k in range(args.num_seeds):
            depth = predict_ours(model, cfg, s, 42 + i + 1000 * k, device)
            per_seed[k].append(score(s, depth, np.broadcast_to(s["gt_mask"][0], depth.shape)))
        print(
            f"[3dfront] {i + 1}/{len(samples)} {s['stem']} AbsRel(seed 0)={per_seed[0][-1]['abs_rel']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    seed_metrics = [aggregate(rows) for rows in per_seed]
    mean = {k: float(np.mean([m[k] for m in seed_metrics])) for k in KEYS}
    std = {k: float(np.std([m[k] for m in seed_metrics])) for k in KEYS}
    lines = [
        "| Row | " + " | ".join(KEYS) + " |",
        "|---|" + "---:|" * len(KEYS),
        fmt_row("paper WT-S", PAPER_WT_S),
    ]
    lines += [fmt_row(f"seed {k}", [m[key] for key in KEYS]) for k, m in enumerate(seed_metrics)]
    lines += [
        fmt_row(f"mean of {args.num_seeds}", [mean[k] for k in KEYS]),
        fmt_row(f"std of {args.num_seeds}", [std[k] for k in KEYS]),
    ]
    print("\n".join(lines))
    Path(args.out).write_text(
        json.dumps(
            {
                "ckpt": args.ckpt,
                "config": args.config,
                "num_seeds": args.num_seeds,
                "mean": mean,
                "std": std,
                "per_seed": seed_metrics,
            },
            indent=1,
        )
    )
    print(f"[3dfront] wrote {args.out}")


if __name__ == "__main__":
    main()
