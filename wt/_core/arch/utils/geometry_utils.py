"""Tiny geometry helper used by the release inference path.

We only need :func:`invert_se3` so it is implemented inline against
``torch`` only.
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


def invert_se3(transform: Float[Tensor, "... 4 4"]) -> Float[Tensor, "... 4 4"]:
    """Invert an SE3 transformation (T -> T^-1)."""
    is_tensor = torch.is_tensor(transform)
    if not is_tensor:
        transform = torch.tensor(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse_rotation = rotation.transpose(-1, -2)
    inverse_translation = -(inverse_rotation @ translation[..., None]).squeeze(-1)
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = inverse_rotation
    inverse[..., :3, 3] = inverse_translation
    inverse[..., 3, 3] = 1.0
    return inverse if is_tensor else inverse.numpy()
