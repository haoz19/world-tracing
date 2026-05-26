"""Inference-only shim for ``wlt.components.nnn``.

The original module is a thin wrapper around ``torch.nn`` that exposes extra
training-time hyperparameters (lr, weight_decay, muon, fp8).  For pure
inference we only need the underlying ``nn.Linear`` / ``nn.LayerNorm``
behaviour; the extra kwargs are silently accepted and dropped so that loading
state-dicts produced by the original ``nnn.Linear`` modules works unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


def _strip_kwargs(kwargs: dict) -> dict:
    for k in (
        "lr",
        "weight_decay",
        "use_muon",
        "out_dtype",
        "weight_init",
        "bias_init",
        "scaling_method",
    ):
        kwargs.pop(k, None)
    return kwargs


class Linear(nn.Linear):
    """Drop-in replacement for ``wlt.components.nnn.Linear`` (inference-only)."""

    def __init__(self, *args, **kwargs):
        kwargs = _strip_kwargs(dict(kwargs))
        super().__init__(*args, **kwargs)


class LayerNorm(nn.LayerNorm):
    """Drop-in replacement for ``wlt.components.nnn.LayerNorm`` (inference-only)."""

    def __init__(self, *args, **kwargs):
        kwargs = _strip_kwargs(dict(kwargs))
        super().__init__(*args, **kwargs)


class Float8Linear(Linear):
    """Stub. We never run fp8 inference in the released code path."""


def _linear_flops(input: torch.Tensor, in_features: int, out_features: int) -> int:
    return int(input.numel() * out_features)
