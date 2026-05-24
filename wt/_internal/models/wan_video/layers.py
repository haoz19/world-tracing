"""Minimal inference-only shim of ``wlt.models.wan_video.layers``.

The original module is ~600 lines and pulls in flash-attention, kv-caching,
pytree, and various Wan2.1 utilities that are only relevant for the Wan
video model itself.  Our multilayer-depth model only uses two helpers:

* :func:`wan_init_linear` -- linear weight init following Wan2.1
* :func:`sinusoidal_embedding_1d` -- 1D sinusoidal positional embedding for
  diffusion timesteps

so we reimplement just those here.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn


def wan_init_linear(m: nn.Linear) -> None:
    """Initialize a linear layer following Wan2.1's recipe."""
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def sinusoidal_embedding_1d(
    dim: int, position: Float[Tensor, "b"], theta: float = 10_000
) -> Float[Tensor, "b d"]:
    """Generate sinusoidal embedding for a 1D position vector.

    Returns ``[cos(omega_k t), sin(omega_k t)]`` along the channel axis,
    where ``omega_k = theta ** (-k / (dim/2))``.
    """
    assert dim % 2 == 0, "dim must be even for sinusoidal embedding!"
    half = dim // 2
    position = position.type(torch.float64)
    freqs = torch.outer(
        position, torch.pow(theta, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=1)
    return x
