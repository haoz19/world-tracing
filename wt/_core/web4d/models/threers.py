"""Inference-only stub of ``wlt.web4d.models.threers``.

The full ``threers.Threers`` model (1700 lines of training-time Gaussian
splatting code, MoGe encoder hooks, camera heads, etc.) is **not** used by
the released multilayer-geometry inference path.  We only re-export
``DecoderBlockSA`` so that ``model.py`` can pick between
``blocks.DecoderBlockDiT`` (head_timestep=True, our default) and the older
``DecoderBlockSA`` (head_timestep=False).
"""

from __future__ import annotations

from wt._core.web4d.models.blocks import DecoderBlockSA

__all__ = ["DecoderBlockSA"]
