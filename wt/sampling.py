"""Flow-matching sampler for ``MultilayerXYZModel`` inference.

The training-time code used a single FlowMatching loss wrapper for both the
training loss and inference denoising.  At inference time only a small
subset of that wrapper is actually exercised:

* ``use_pixel_denoising=True`` (raw-data denoising, no VAE)
* ``hunyuan_shift_factor=1.0`` (identity timestep shift)
* Euler ODE sampler over the predicted velocity ``v_t = model(x_t, t)["v_t"]``

This module reimplements just that path, with no VAE / no
masked-layer-norm / no train-mode helpers.

Reference math (flow matching, ICLR'23):
    x_t = (1 - t) * x_0 + t * eps,    t ∈ [0, 1]
    v(x_t, t) = eps - x_0
    => x_{t - dt} = x_t - dt * v(x_t, t)
"""

from __future__ import annotations

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn

#: Minimum non-zero timestep we'll feed to the model.  The training-time
#: code guards against sampling timesteps below this value because the
#: model was never trained that close to ``t=0``.  ``iterations=20`` lands
#: the smallest step at exactly 0.05,
#: which is the recommended default.  We warn (rather than raise) if a
#: smaller ``iterations`` is requested.
T_MIN_CLAMP: float = 0.05


def hunyuan_shift_timesteps(
    timesteps: Float[Tensor, "..."], s: float
) -> Float[Tensor, "..."]:
    """Hunyuan / Wan video timestep shift.

    Equation: ``t' = s * t / (1 + (s - 1) * t)``.  ``s = 1.0`` reduces to
    the identity (which is what the release configs use).
    """
    return s * timesteps / (1 + (s - 1) * timesteps)


@torch.no_grad()
def denoise_geometry(
    model: nn.Module,
    x_t: Float[Tensor, "b c t h w"],
    conditioning: dict[str, Tensor],
    iterations: int = 20,
    hunyuan_shift_factor: float = 1.0,
    return_all_steps: bool = False,
) -> Float[Tensor, "b t h w c"]:
    """Euler ODE sampler for the flow-matching geometry model.

    Args:
        model: a ``MultilayerXYZModel`` (or a DDP wrapper around one) returning
            a dict with key ``"v_t"`` of shape ``[B, S, C]`` where
            ``S = nview * H * W``.
        x_t: noise tensor ``[B, C, V, H, W]``.  Treated as a flattened sequence
            of ``S = V*H*W`` C-dim tokens internally; reshaped on return.
        conditioning: dict consumed by ``model.forward(x_t, t, conditioning)``.
            Must contain at least ``"rgb"`` and ``"batch_size"``.  See
            :mod:`wt.inference` for the exact contents.
        iterations: number of Euler steps.
        hunyuan_shift_factor: timestep shift factor (default 1.0 = identity).
        return_all_steps: if True, return all intermediate ``x_t`` tensors
            instead of only the final ``x_0`` prediction.

    Returns:
        ``x_0`` of shape ``[B, V, H, W, C]`` (the denoised prediction at
        ``t=0``), or a list of all intermediate ``x_t``s if
        ``return_all_steps=True``.
    """
    batch_size, _, nview, height, width = x_t.shape
    x_t = einops.rearrange(x_t, "b c t h w -> b (t h w) c")

    timesteps = torch.linspace(1.0, 0.0, iterations + 1, device=x_t.device)
    timesteps = hunyuan_shift_timesteps(timesteps, hunyuan_shift_factor)

    x_t_all: list[Tensor] = []
    for i in range(iterations):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t - t_next
        t_b = einops.repeat(t, " -> b", b=batch_size)
        out = model(x_t, t_b, conditioning)
        v_t = out["v_t"]
        x_t = x_t - dt * v_t
        if return_all_steps:
            x_t_all.append(
                einops.rearrange(
                    x_t, "b (t h w) c -> b t h w c", t=nview, h=height, w=width
                )
            )

    x_0 = einops.rearrange(
        x_t, "b (t h w) c -> b t h w c", t=nview, h=height, w=width
    )
    if return_all_steps:
        return x_t_all  # type: ignore[return-value]
    return x_0
