# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import math

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from wt._core.math import safe_math
from wt._core.utils import torch_utils


def contract(x):
    """Contracts points towards the origin (Eq 10 of arxiv.org/abs/2111.12077)."""
    # Clamping to 1 produces correct scale inside |x| < 1
    x_mag_sq = torch.maximum(torch.tensor(1.0), torch.sum(x**2, dim=-1, keepdim=True))
    scale = (2 * torch.sqrt(x_mag_sq) - 1) / x_mag_sq
    z = scale * x
    return z


def inv_contract(z):
    """The inverse of contract()."""
    # Clamping to 1 produces correct scale inside |z| < 1
    z_mag_sq = torch.maximum(torch.tensor(1.0), torch.sum(z**2, dim=-1, keepdim=True))
    inv_scale = 2 * torch.sqrt(z_mag_sq) - z_mag_sq
    x = z / inv_scale
    return x


def construct_ray_warps(fn, t_near, t_far, *, fn_inv=None):
    """Construct a bijection between metric distances and normalized distances.

    See the text around Equation 11 in https://arxiv.org/abs/2111.12077 for a
    detailed explanation.

    Args:
      fn: the function to ray distances.
      t_near: a tensor of near-plane distances.
      t_far: a tensor of far-plane distances.
      fn_inv: Optional, if not None then it's used as the inverse of fn().

    Returns:
      t_to_u: a function that maps distances to normalized distances in [0, 1].
      u_to_t: the inverse of t_to_u.
    """
    if fn is None:
        fn_fwd = lambda x: x
        fn_inv = lambda x: x
    else:
        fn_fwd = fn
        if fn_inv is None:
            # A simple mapping from some functions to their inverse.
            inv_mapping = {
                "reciprocal": torch.reciprocal,
                "log": torch.exp,
                "exp": torch.log,
                "sqrt": torch.square,
                "square": torch.sqrt,
            }
            fn_inv = inv_mapping[fn.__name__]
    fn_t_near, fn_t_far = (fn_fwd(t) for t in (t_near, t_far))
    # Forcibly clip t to the range of valid values, to guard against inf's.
    t_clip = lambda t: torch.clamp(t, t_near, t_far)
    t_to_u = lambda t: (fn_fwd(t_clip(t)) - fn_t_near) / (fn_t_far - fn_t_near)
    u_to_t = lambda u: t_clip(fn_inv(u * fn_t_far + (1 - u) * fn_t_near))
    return t_to_u, u_to_t


def piecewise_warp_fwd(x: torch.Tensor, eps: float = safe_math.EPS):
    """A piecewise combo of linear and reciprocal to allow t_near=0."""
    return torch.where(x < 1, 0.5 * x, 1 - 0.5 / x.clamp(min=eps))


def piecewise_warp_inv(x: torch.Tensor, eps: float = safe_math.EPS):
    """The inverse of `piecewise_warp_fwd`."""
    return torch.where(x < 0.5, 2 * x, 0.5 / (1 - x).clamp(min=eps))


def power_ladder(x, p, premult=None, postmult=None):
    """Tukey's power ladder, with a +1 on x, some scaling, and special cases."""
    # Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
    if premult is not None:
        x = x * premult
    p = torch.tensor(p)
    xp = torch.abs(x)
    xs = xp / torch.abs(p - 1).clamp(min=safe_math.TINY)
    y = torch.sign(x) * torch.abs(p - 1) / p * ((xs + 1) ** p - 1)
    if postmult is not None:
        y = y * postmult
    return y


def inv_power_ladder(y, p, premult=None, postmult=None):
    """The inverse of `power_ladder()`."""
    if postmult is not None:
        y /= postmult
    p = torch.tensor(p)
    yp = torch.abs(y)
    x = (
        torch.sign(y)
        * torch.abs(p - 1)
        * (((p / torch.abs(p - 1)) * yp + 1) ** (1 / p) - 1)
    )
    if premult is not None:
        x /= premult
    return x


def get_default_ray_warps(t_near, t_far, p: float = -1.5, premult: float = 2):
    fn_fwd = functools.partial(power_ladder, p=p, premult=premult)
    fn_inv = functools.partial(inv_power_ladder, p=p, premult=premult)
    return construct_ray_warps(fn_fwd, t_near, t_far, fn_inv=fn_inv)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in NeRF paper."""

    def __init__(self, frequencies: int, append_id: bool = True):
        """
        Args:
            frequencies: The number of frequency bands to use.
            append_id: If true, append the input to the output.
        """
        super().__init__()
        self.f = nn.Buffer(torch.pi * (2.0 ** torch.arange(frequencies)))
        self.append_id = append_id

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (..., d_in).

        Returns:
            e: Tensor of shape (..., d_out), where:
                d_out = 2 * freq * d_in + d_in if append_id == True;
                d_out = 2 * freq * d_in if append_id == False
        """
        in_shape = x.shape
        freq, d_in = self.f.shape[0], in_shape[-1]
        x = x.view(-1, d_in)
        xf = (self.f * x[..., None]).view(-1, d_in * freq)
        out = [xf.sin(), xf.cos()]
        if self.append_id:
            out.append(x)
        out = torch.cat(out, dim=-1)
        out = out.view(*in_shape[:-1], -1)
        return out


def construct_perp_basis(directions):
    """Construct a perpendicular basis for each 3-vector in `directions`."""
    *batch_shape, _ = directions.shape
    if directions.shape[-1] != 3:
        raise ValueError(f"directions must be 3D, but is {directions.shape[-1]}D")

    # To generate a vector perpendicular to `directions`, we take a cross-product
    # with an arbitrary vector [0, 0, 1].
    perp_vec = torch_utils.tensor_like(directions, [0.0, 0.0, 1.0])
    perp_vec = perp_vec.view(*[1 for _ in batch_shape], 3)
    cross1a = torch.linalg.cross(directions, perp_vec)

    # In the rare case that `directions` is very close to [0, 0, 1], we compute an
    # alternate cross-product with [1, 1, 1] to use instead.
    perp_vec = torch_utils.tensor_like(directions, [1.0, 1.0, 1.0])
    perp_vec = perp_vec.view(*[1 for _ in batch_shape], 3)
    cross1b = torch.linalg.cross(directions, perp_vec)
    use_b = torch.all(torch.abs(cross1a) < torch.finfo(torch.float32).eps, axis=-1)
    cross1 = torch.where(use_b[..., None], cross1b, cross1a)

    # Crossing `directions` with `cross1` gives us our 3rd vector.
    cross2 = torch.linalg.cross(directions, cross1)

    return F.normalize(cross1, dim=-1), F.normalize(cross2, dim=-1)


def hexify(
    origins: Float[Tensor, "*n 3"],
    directions: Float[Tensor, "*n 3"],
    radii: Float[Tensor, "*n 1"],
    t0: Float[Tensor, "*n 1"],
    t1: Float[Tensor, "*n 1"],
    randomize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce hexagon-shaped samples from ray segments."""
    device = origins.device
    t0 = t0.squeeze(-1)
    t1 = t1.squeeze(-1)

    # Construct a base set of angles, by linspacing [0, 2pi] in a specific order.
    # This is one of two orderings of angles that doesn't induce any anisotropy
    # into the sample covariance of the multisample coordinates. Any rotation and
    # mirroring along the z-axis of this ordering is also valid.
    # There exists one alternative valid ordering, which is [0, 3, 2, 5, 4, 1].
    # This seems to work less well though likely because of the strong correlation
    # between adjacent angles.
    thetas = (torch.pi / 3) * torch_utils.tensor_like(origins, [0, 2, 4, 3, 5, 1])

    # Lift the angles to the size of the rays.
    sz = (*t0.shape, len(thetas))
    thetas = torch.broadcast_to(thetas, sz)

    if randomize:
        # Randomly reverse the order of half of the hexes.
        flip = torch.bernoulli(torch.full(sz[:-1], 0.5, device=device)).bool()
        thetas = torch.where(flip[..., None], torch.flip(thetas, dims=[-1]), thetas)
        # Rotate each hex by some random amount.
        thetas += (2 * torch.pi) * torch.rand(sz[:-1], device=device)[..., None]
    else:
        # If we're deterministic, flip and shift every other hex by 30 degrees.
        flip = (torch.arange(thetas.shape[-2], device=device) % 2).bool()
        thetas = torch.where(flip[..., None], torch.flip(thetas, dims=[-1]), thetas)
        thetas += (flip * torch.pi / 6)[..., None]

    # TODO(barron): Plumb through the dx/dy frame for the original ray in the
    # image plane, to avoid the need of this.
    perp_axis1, perp_axis2 = construct_perp_basis(directions)

    # Grab each t-interval's midpoint and half-width.
    s = (t0 + t1) / 2
    d = (t1 - t0) / 2

    # Compute the length along the ray for each multisample, using mip-NeRF math.
    cz = t0[..., None] + safe_math.safe_div(d, (d**2 + 3 * s**2))[..., None] * (
        (t1**2 + 2 * s**2)[..., None]
        + (3 / math.sqrt(7))
        * (torch.arange(6, device=device) * (2 / 5) - 1)
        * safe_math.safe_sqrt((d**2 - s**2) ** 2 + 4 * s**4)[..., None]
    )

    # Compute the offset from the ray for each multisample.
    perp_mag = math.sqrt(0.5) * radii * cz

    # Go from ray coordinate to world coordinates.
    cx = perp_mag * torch.cos(thetas)
    cy = perp_mag * torch.sin(thetas)
    control = (
        origins[..., None, :]
        + perp_axis1[..., None, :] * cx[..., None]
        + perp_axis2[..., None, :] * cy[..., None]
        + directions[..., None, :] * cz[..., None]
    )

    return control, perp_mag


def contract3_isoscale(x: torch.Tensor) -> torch.Tensor:
    """A fast version of track_isotropic(contract, *)'s scaling for 3D inputs."""
    if x.shape[-1] != 3:
        raise ValueError(f"Inputs must be 3D, are {x.shape[-1]}D.")
    norm_sq = torch.sum(x**2, axis=-1).clamp(min=1.0)
    # Equivalent to cbrt((2 * sqrt(norm_sq) - 1) ** 2) / norm_sq:
    return torch.exp(
        2 / 3 * torch.log(2 * torch.sqrt(norm_sq) - 1) - torch.log(norm_sq)
    )
