"""Checkpoint loading helpers shared by the ``examples/`` scripts.

The release ships three model configs:

* ``r75b`` -- :class:`MultilayerXYZModel` for static objects (1.7B params,
  ``image_size=504``).
* ``r69e`` -- multilayer-geometry scene model (1.5B params, ``image_size=504``).
* ``r76``  -- dynamic-object video model with temporal attention (2.1B
  params, ``image_size=336``).

Each config consists of:

* ``model_kwargs``     -- passed verbatim to ``MultilayerXYZModel(**kwargs)``;
* ``image_size``       -- input resolution the checkpoint was trained at;
* ``inference_kwargs`` -- defaults passed to
  :func:`wt.inference.inference_diffusion` (or its video / multi-view
  counterparts).

Checkpoint files are hosted on Hugging Face Hub.  ``--ckpt`` accepts either
a local ``.pt`` path or a Hugging Face shorthand:

* ``hf://repo_id``                       (uses ``model.pt`` from that repo)
* ``hf://repo_id/relative/path.pt``      (explicit file)

You can also pass the bare config name (``r75b`` / ``r69e`` / ``r76``) and we
will resolve it via :data:`HF_REPOS` below.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Hugging Face Hub mapping
# ---------------------------------------------------------------------------


HF_REPOS = {
    "r75b": "haoz19/object-model-6layer",
    "r69e": "haoz19/scene-model-6layer",
    "r69g": "haoz19/scene-model-6layer",
    "r76":  "haoz19/dynamic-model-16frame",
}
"""Map model config name → public Hugging Face repository.

The actual checkpoint files inside each repo are named ``model.pt`` (a
plain ``torch.save`` of either ``state_dict`` or ``{"model_state_dict":
..., "ema_state_dict": ...}``).
"""

HF_DEFAULT_FILENAME = "model.pt"


def _resolve_hf_path(path_or_uri: str) -> tuple[str, str] | None:
    """Parse ``hf://repo_id[/filename]`` or a bare config name.

    Returns ``(repo_id, filename)`` if ``path_or_uri`` is an HF shorthand,
    otherwise ``None``.
    """
    if isinstance(path_or_uri, Path):
        return None
    s = str(path_or_uri).strip()
    if s in HF_REPOS:
        return HF_REPOS[s], HF_DEFAULT_FILENAME
    if s.startswith("hf://"):
        rest = s[len("hf://"):]
        parts = rest.split("/", 2)
        if len(parts) >= 2:
            repo_id = "/".join(parts[:2])
            fname = "/".join(parts[2:]) if len(parts) > 2 else HF_DEFAULT_FILENAME
            return repo_id, fname or HF_DEFAULT_FILENAME
    return None


def download_from_hf(repo_id: str, filename: str = HF_DEFAULT_FILENAME,
                     cache_dir: str | os.PathLike | None = None) -> str:
    """Download a checkpoint file from Hugging Face Hub.

    Parameters
    ----------
    repo_id
        e.g. ``"haoz19/object-model-6layer"``.
    filename
        Relative file path inside the repo.  Defaults to ``model.pt``.
    cache_dir
        Local cache directory.  ``None`` falls back to the
        ``HF_HOME`` / ``~/.cache/huggingface`` default.

    Returns the local path to the (possibly cached) file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download checkpoints from HF Hub. "
            "Install it with `pip install huggingface_hub` (or `pip install wt`)."
        ) from e
    return hf_hub_download(
        repo_id=repo_id, filename=filename, cache_dir=cache_dir
    )


# ---------------------------------------------------------------------------
# Model configs (copied verbatim from the experiment registry).
# ---------------------------------------------------------------------------


_R75B_KWARGS = dict(
    num_layers=6,
    decoder_embed_dim=1536,
    num_decoder_blocks=42,
    num_decoder_heads=24,
    num_head_blocks=6,
    use_rope=True,
    ls_init_values=1e-4,
    layer_embed_mode="film",
    head_film=True,
    head_timestep=True,
    freeze_encoder=True,
    depth_only=True,
    cfm_mask=False,
    cfm_noise_type="fixed_0.5",
    predict_color=False,
    use_ar=False,
    use_cross_attn=False,
    use_split_head=True,
    split_head_mode="transformer",
    model_task="split_token",
    output_mode="xyz",
    patch_size=14,
    head_attn_mode="frame_ray_global",
    attention_pattern="frame_ray_global",
)


_R69E_KWARGS = dict(
    num_layers=6,
    decoder_embed_dim=1536,
    num_decoder_blocks=36,
    num_decoder_heads=24,
    num_head_blocks=6,
    use_rope=True,
    ls_init_values=1e-4,
    layer_embed_mode="film_adaln",
    head_film=True,
    head_timestep=True,
    freeze_encoder=True,
    depth_only=True,
    cfm_mask=False,
    cfm_noise_type="fixed_0.5",
    predict_color=False,
    use_ar=False,
    use_cross_attn=False,
    use_split_head=True,
    split_head_mode="transformer",
    model_task="split_token",
    output_mode="xyz",
    patch_size=14,
    head_attn_mode="frame_ray_global",
    attention_pattern="frame_ray_global",
)


_R76_KWARGS = dict(
    **_R75B_KWARGS,
    use_temporal_blocks=True,
    temporal_ls_init=1e-5,
    temporal_mlp_ratio=4.0,
    temporal_use_qk_norm=True,
)


CONFIGS = {
    "r75b": {
        "model_kwargs": _R75B_KWARGS,
        "image_size": 504,
        "inference_kwargs": dict(
            num_steps=20,
            xyz_norm_mode="zscore",
            output_mode="xyz",
            model_task="split_token",
            depth_only=True,
        ),
    },
    "r69e": {
        "model_kwargs": _R69E_KWARGS,
        "image_size": 504,
        "inference_kwargs": dict(
            num_steps=20,
            xyz_norm_mode="median_log",
            output_mode="xyz",
            model_task="split_token",
            depth_only=True,
        ),
    },
    "r69g": {
        # Same architecture as r69e (warm-continued from r69e latest).
        # The only training-time delta is `xyz_norm_mode=median_log_global`
        # (per-batch global rescale instead of per-sample median_log).
        "model_kwargs": _R69E_KWARGS,
        "image_size": 504,
        "inference_kwargs": dict(
            num_steps=20,
            xyz_norm_mode="median_log_global",
            output_mode="xyz",
            model_task="split_token",
            depth_only=True,
        ),
    },
    "r76": {
        "model_kwargs": _R76_KWARGS,
        "image_size": 336,
        "inference_kwargs": dict(
            num_steps=20,
            xyz_norm_mode="zscore",
            output_mode="xyz",
            model_task="split_token",
            depth_only=True,
        ),
    },
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(ckpt_path: str | Path, model: torch.nn.Module) -> dict:
    """Load a ``MultilayerXYZModel`` checkpoint into ``model``.

    Handles two common formats produced by the training pipeline:

    * Raw ``model_state_dict`` (i.e. ``model.state_dict()``).
    * ``checkpoint["model_state_dict"]`` /
      ``checkpoint["ema_state_dict"]`` from the training loop.  EMA is
      preferred if present.

    Also normalises the ``_orig_mod.`` prefix difference that arises from
    ``torch.compile``: if the checkpoint and the live model disagree on the
    prefix, one side is rewritten so the keys line up.
    """
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        ema = checkpoint.get("ema_state_dict")
        state = ema if ema is not None else checkpoint["model_state_dict"]
    else:
        state = checkpoint
        checkpoint = {"model_state_dict": state}

    _PFX = "_orig_mod."
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    if state_keys and model_keys and not (state_keys & model_keys):
        m_has = all(k.startswith(_PFX) for k in model_keys)
        s_has = all(k.startswith(_PFX) for k in state_keys)
        if m_has and not s_has:
            state = {f"{_PFX}{k}": v for k, v in state.items()}
        elif s_has and not m_has:
            state = {k[len(_PFX):]: v for k, v in state.items()}

    for key in list(state.keys()):
        if key in model.state_dict() and state[key].shape != model.state_dict()[key].shape:
            print(f"  shape mismatch, dropping: {key}")
            del state[key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}):", missing[:5], "...")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}):", unexpected[:5], "...")
    return checkpoint


def resolve_ckpt_path(ckpt_path: str | Path) -> str:
    """Resolve ``ckpt_path`` to a local file path.

    Accepts:
      * a local filesystem path (returned as-is once verified);
      * an HF shorthand ``hf://repo_id[/file.pt]``;
      * a bare config name ``r75b`` / ``r69e`` / ``r76`` (looked up in
        :data:`HF_REPOS`).

    Downloads the file from Hugging Face Hub on first use; subsequent calls
    return the cached local path.
    """
    hf = _resolve_hf_path(ckpt_path)
    if hf is not None:
        repo_id, filename = hf
        print(f"[wt] resolving HF checkpoint: {repo_id}/{filename}")
        local = download_from_hf(repo_id, filename)
        print(f"[wt]   -> {local}")
        return local
    p = Path(str(ckpt_path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path!s}. "
            f"Pass an HF shorthand (e.g. 'hf://haoz19/object-model-6layer') "
            f"or a config name ('r75b' / 'r69e' / 'r76') to download "
            f"the released weights from Hugging Face Hub instead."
        )
    return str(p)


def build_model_and_load_ckpt(
    config_name: str,
    ckpt_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Convenience wrapper: build the chosen config, load the checkpoint, eval().

    ``ckpt_path`` may be a local ``.pt`` file, an ``hf://`` shorthand, or a
    config name -- see :func:`resolve_ckpt_path`.
    """
    from wt.model import MultilayerXYZModel

    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config {config_name!r}; available: {sorted(CONFIGS)}"
        )
    cfg = CONFIGS[config_name]
    model = MultilayerXYZModel(**cfg["model_kwargs"]).to(device)
    print(
        f"[wt] {config_name}: "
        f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    local_ckpt = resolve_ckpt_path(ckpt_path)
    print(f"[wt] loading checkpoint: {local_ckpt}")
    load_checkpoint(local_ckpt, model)
    model.eval()
    return model, cfg
