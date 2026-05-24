"""Inference-only stub of ``wlt.web4d.models.head_utils``.

The original module is ~1000 lines of training-time camera heads, voxel
prediction heads and DPT depth heads.  Our released configs (r75b, r69e, r76)
all run with ``use_camera_head=False`` so this whole module is unreachable.
We expose ``MLPCameraHead`` as a stub that raises if anyone tries to use it.
"""

from __future__ import annotations

from torch import nn


class MLPCameraHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(
            "MLPCameraHead is a training-only camera head.  "
            "Release configs (r75b/r69e/r76) all use use_camera_head=False."
        )
