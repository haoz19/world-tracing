"""Lightweight file-IO helpers used by the vendored MoGe loader.

Only ``cache_remote_path`` is referenced (by vendored MoGe's
``from_pretrained``).  The released inference path does NOT call
``MoGeModel.from_pretrained`` -- we instantiate ``MoGeModel`` with random
weights and then load our own checkpoint over it -- so this code path is
unreachable.  We still expose a stub so the imports succeed.
"""

from __future__ import annotations

import pathlib
from typing import Any


def cache_remote_path(path: Any, **_kwargs) -> pathlib.Path:
    raise NotImplementedError(
        "wt._core.file_io.cache_remote_path is an inference-only stub.  "
        "If you reached this error you are probably calling "
        "MoGeModel.from_pretrained(); the release path instantiates "
        "MoGeModel with random weights and loads its own state dict instead."
    )


def open(path, mode: str = "r"):  # noqa: A001
    return pathlib.Path(path).open(mode)
