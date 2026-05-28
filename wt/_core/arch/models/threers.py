"""Inference-only stub of the legacy backbone.

The full training-time backbone (~1700 lines of Gaussian splatting code,
MoGe encoder hooks, camera heads, etc.) is **not** used by the released
multilayer-geometry inference path.  We only re-export ``DecoderBlockSA``
so that ``model.py`` can pick between ``blocks.DecoderBlockDiT``
(head_timestep=True, our default) and the older ``DecoderBlockSA``
(head_timestep=False).
"""

from __future__ import annotations

from wt._core.arch.models.blocks import DecoderBlockSA

__all__ = ["DecoderBlockSA"]
