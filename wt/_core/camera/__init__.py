"""Minimal pin-hole camera placeholder used only as a type annotation.

The released multilayer-geometry inference path never builds or consumes a
``Camera`` object (we pass ``camera=None`` everywhere, which is the default
for the r75b / r69e / r76 configs), so we only need to make ``Camera``
resolvable as a type annotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    """Placeholder dataclass used purely as a type annotation.

    ``MultilayerXYZModel`` and the underlying ``MultilayerBackbone`` only consume
    ``Camera`` objects when ``use_raymap=True`` or ``use_camera_head=True``,
    neither of which is enabled in the release configs.
    """

    camera_to_world: torch.Tensor | None = None
    camera_to_pixel: torch.Tensor | None = None
    image_size_xy: torch.Tensor | None = None


def update_extrinsic(camera: Camera, position=None, **_kwargs) -> Camera:
    raise NotImplementedError(
        "camera.update_extrinsic is not used by the release inference path."
    )


def resize_with_crop(camera: Camera, target_size) -> Camera:  # pragma: no cover
    raise NotImplementedError(
        "camera.resize_with_crop is not used by the release inference path."
    )
