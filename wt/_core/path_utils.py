"""Lightweight path helpers used by the vendored MoGe loader.

Only ``ensure_pathlike`` is used at this layer (by vendored MoGe) — and the
inference release never exercises the GCS/S3 download paths (we always
instantiate ``MoGeModel`` with random weights and then load our own
checkpoint).  This stub just normalises strings to ``pathlib.Path``.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any


def ensure_pathlike(path: Any, **_kwargs) -> pathlib.Path:
    if isinstance(path, os.PathLike):
        return path  # type: ignore[return-value]
    if isinstance(path, (str, bytes)):
        if isinstance(path, bytes):
            path = path.decode()
        return pathlib.Path(path)
    raise ValueError(f"Input {path=} is not PathLike!")


class WPath(pathlib.PosixPath):
    """Path subclass kept for source compatibility with older import sites."""


__all__ = ["ensure_pathlike", "WPath"]
