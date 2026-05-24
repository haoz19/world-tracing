"""CLI helpers shared by the ``examples/`` scripts.

Keeps the ``argparse`` boilerplate out of each example so the per-script
files focus on the actual pipeline (preprocess → forward → visualise).
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SWEEP_BASE_SEED = 42
DEFAULT_SWEEP_SIZE = 4


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Translate ``args.seed`` / ``args.num_seeds`` into a concrete seed list.

    Semantics:

    * Neither flag set -> ``[42, 43, 44, 45]`` (the default 4-seed sweep).
    * ``--seed N`` only -> ``[N]`` (single deterministic sample).
    * ``--num-seeds K`` only -> ``[42, 43, ..., 42 + K - 1]``.
    * Both ``--seed N`` and ``--num-seeds K`` -> ``[N, N+1, ..., N+K-1]``.

    Use this helper in every example script that drives ``inference_diffusion``
    so the multi-seed UX stays consistent across object / scene / video /
    textured-mesh pipelines.
    """
    seed = getattr(args, "seed", None)
    num_seeds = getattr(args, "num_seeds", None)
    if seed is None and num_seeds is None:
        return [
            DEFAULT_SWEEP_BASE_SEED + i for i in range(DEFAULT_SWEEP_SIZE)
        ]
    if num_seeds is None:
        return [int(seed)]
    if num_seeds < 1:
        raise SystemExit("--num-seeds must be >= 1")
    base = int(seed) if seed is not None else DEFAULT_SWEEP_BASE_SEED
    return [base + i for i in range(int(num_seeds))]


def parse_bg_color(raw: str) -> tuple[int, int, int] | None:
    """Parse ``--bg-color`` value.

    Accepts:
      * ``"none"`` (case-insensitive) → ``None`` (skip the override)
      * ``"R,G,B"`` with three integers in ``[0, 255]``
    """
    if raw is None:
        return (0, 0, 0)
    if raw.lower() == "none":
        return None
    try:
        parts = [int(x) for x in raw.split(",")]
    except ValueError:
        raise SystemExit(
            "--bg-color must be three comma-separated integers in [0,255] "
            f"or 'none'; got {raw!r}"
        )
    if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
        raise SystemExit(
            "--bg-color must be three integers in [0,255] separated by "
            f"commas; got {raw!r}"
        )
    return tuple(parts)  # type: ignore[return-value]


def add_common_args(p: argparse.ArgumentParser, default_out: str) -> None:
    """Register the args that every example shares (ckpt, config, seed, ...)."""
    from wt.checkpoint import CONFIGS

    p.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help=(
            "Checkpoint -- one of: a local path to a .pt / .safetensors "
            "file, an HF shorthand ``hf://owner/repo[/file.pt]``, or a "
            "bare config name (``r75b``, ``r69e``, ``r76``) which "
            "downloads the released weights from Hugging Face Hub."
        ),
    )
    p.add_argument(
        "--config",
        choices=sorted(CONFIGS.keys()),
        default="r75b",
        help="Model config to instantiate (default: r75b)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(default_out),
        help="Output .rrd file",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run a single deterministic seed.  Mutually exclusive with the "
            "default multi-seed behaviour: when neither ``--seed`` nor "
            "``--num-seeds`` is set, the script runs ``--num-seeds 4`` "
            "(seeds ``42, 43, 44, 45``) so you can eyeball the four "
            "diffusion samples side-by-side and pick the best.  "
            "Combine with ``--num-seeds K`` to run ``K`` seeds starting "
            "at the given ``--seed`` value."
        ),
    )
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help=(
            "Number of independent diffusion seeds to sample (default "
            "behaviour: 4 when neither ``--seed`` nor ``--num-seeds`` is "
            "specified).  Pass ``--num-seeds 1`` for the fastest "
            "single-sample mode."
        ),
    )
    p.add_argument(
        "--alpha-erode",
        type=int,
        default=0,
        help=(
            "Erode the alpha mask by N pixels.  Default 0 (no erode), matching "
            "the preprocessing used for the video-selection inference runs.  "
            "Bump to 3-8 only for noisy SAM/matting cutouts."
        ),
    )
    p.add_argument(
        "--no-center-crop",
        action="store_true",
        help="Skip object-centred crop (use this for already-framed images).",
    )
    p.add_argument(
        "--crop-ratio",
        type=float,
        default=2.0 / 3.0,
        help=(
            "Object-to-crop ratio for the centred crop.  The object's longest "
            "axis is sized to occupy this fraction of the square crop side; "
            "the rest is empty margin (zero-padded if it extends past the "
            "image boundary).  Default 2/3 (~0.667), matching the framing "
            "used during training (Objaverse renders).  Use higher (e.g. "
            "0.8) for tighter framing."
        ),
    )
    p.add_argument(
        "--bg-color",
        type=str,
        default="128,128,128",
        help=(
            "RGB triple (0-255) that the input RGB is **alpha-blended** "
            "against before being fed to the model: "
            "``rgb_out = rgb * alpha + bg * (1 - alpha)``.  This preserves "
            "soft cutout edges instead of binary-painting them.  Defaults "
            "to mid-gray ``128,128,128`` (matches the video-selection run "
            "preprocessing).  Pass ``none`` to skip the blend and feed the "
            "raw RGB to the encoder (use this when the input is already "
            "pre-composited against the desired background)."
        ),
    )
    p.add_argument(
        "--no-auto-alpha",
        action="store_true",
        help=(
            "Disable the near-white-background auto-alpha heuristic.  When an "
            "input image has no alpha channel, by default a binary alpha is "
            "derived from the near-white pixels (useful for SAM/SDXL "
            "outputs).  Pass this flag to skip the heuristic and treat the "
            "entire image as foreground."
        ),
    )
    p.add_argument(
        "--layer-timeline",
        action="store_true",
        help=(
            "Log the prediction along a ``layer`` timeline so the viewer can "
            "scrub through the layers one at a time, revealing the occluded "
            "geometry behind each preceding layer."
        ),
    )
