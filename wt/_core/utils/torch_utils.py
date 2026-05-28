"""Small ``torch`` helpers used by the released inference path."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from torch import Tensor


def tensor_like(reference: Tensor, target: Any, **kwargs) -> Tensor:
    kwargs["dtype"] = kwargs.get("dtype", reference.dtype)
    kwargs["device"] = kwargs.get("device", reference.device)
    return torch.tensor(target, **kwargs)


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@contextlib.contextmanager
def maybe_autocast(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """Wrap ``torch.autocast``, no-op'ing on CPU+float32 (which autocast cannot handle)."""
    assert dtype in (torch.float16, torch.bfloat16, torch.float32), dtype
    assert device.type in ("cpu", "cuda"), device
    if device.type == "cpu" and dtype == torch.float32:
        with contextlib.nullcontext():
            yield
    else:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield
