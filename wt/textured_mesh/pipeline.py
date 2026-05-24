"""TRELLIS.2 hybrid driver — inject our voxel coords into stages 2+3.

This module is the only place that imports ``trellis2`` / ``o_voxel``.  The
import is deferred until you actually call :func:`load_trellis2_pipeline`
so that ``wt`` itself stays usable without the heavy TRELLIS.2 stack.

Install TRELLIS.2 separately (see the project README):

.. code-block:: bash

    git clone https://github.com/microsoft/TRELLIS.2
    cd TRELLIS.2
    bash setup.sh --new-env --basic --flash-attn --o-voxel --nvdiffrast \\
                  --cumesh --flexgemm

Then either ``conda activate trellis2`` before running ``wt`` examples, or
pass ``--trellis2-path /path/to/TRELLIS.2`` so the driver can prepend the
repo to ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _ensure_trellis2_importable(trellis2_path: str | os.PathLike | None) -> None:
    """Prepend the local TRELLIS.2 checkout to ``sys.path`` if needed.

    Does nothing if ``trellis2`` is already importable (e.g. installed
    into the active conda env via ``setup.sh``).
    """
    if trellis2_path is not None:
        p = Path(trellis2_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"--trellis2-path does not exist: {p}.  Did you clone "
                f"https://github.com/microsoft/TRELLIS.2 ?"
            )
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_trellis2_pipeline(
    model_id: str = "microsoft/TRELLIS.2-4B",
    device: str | torch.device = "cuda",
    trellis2_path: str | os.PathLike | None = None,
    attn_backend: str | None = None,
):
    """Load a :class:`Trellis2ImageTo3DPipeline` from Hugging Face.

    Args:
        model_id: HF repo id.  ``microsoft/TRELLIS.2-4B`` is the public
            release (works for the 1024 pipeline) -- if you have a
            different fine-tune locally, pass its path here.
        device: torch device.
        trellis2_path: optional path to a local TRELLIS.2 checkout.  Use
            this if ``trellis2`` is not already on ``sys.path``.
        attn_backend: optional override for ``SPARSE_ATTN_BACKEND``
            (``"flash_attn"`` or ``"xformers"``).  ``None`` keeps the
            TRELLIS.2 default (``flash_attn``).

    Returns:
        The loaded ``Trellis2ImageTo3DPipeline`` instance, already moved
        to ``device``.
    """
    _ensure_trellis2_importable(trellis2_path)
    if attn_backend is not None:
        os.environ["SPARSE_ATTN_BACKEND"] = attn_backend

    try:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Could not import `trellis2`.  Install the public TRELLIS.2 "
            "repo (https://github.com/microsoft/TRELLIS.2) via its "
            "`setup.sh`, then either activate that conda env or pass "
            "`--trellis2-path /path/to/TRELLIS.2`."
        ) from exc

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    pipeline.to(device)
    return pipeline


@torch.no_grad()
def inject_coords_into_trellis2(
    pipeline,
    pil_rgba: Image.Image,
    coords_xyz: np.ndarray,
    seed: int = 42,
    pipeline_type: str = "1024",
    preprocess_image: bool = True,
):
    """Run TRELLIS.2 stages 2+3 with our voxel coords as the Stage-1 output.

    Args:
        pipeline: ``Trellis2ImageTo3DPipeline`` from
            :func:`load_trellis2_pipeline`.
        pil_rgba: input image (RGBA PIL Image).  If
            ``preprocess_image=True`` (default), TRELLIS.2's own
            ``preprocess_image`` (including rembg + crop + recentre) is
            applied; otherwise we pass the image through verbatim (use
            this when your input is already a clean cutout).
        coords_xyz: ``(M, 3) int32`` voxel coords in ``[0, ss_res-1]``
            from :func:`wt.textured_mesh.v4_ray_fill`.  A leading batch
            index column is added internally.
        seed: random seed for the diffusion samplers.
        pipeline_type: one of ``"512"``, ``"1024"``, ``"1024_cascade"``,
            ``"1536_cascade"`` (see TRELLIS.2 README).  Higher resolution
            = better detail, more VRAM / time.

    Returns:
        ``list[MeshWithVoxel]`` -- typically length 1.  Each entry has
        ``.vertices, .faces, .attrs, .coords, .layout, .voxel_size``,
        ready for :func:`save_mesh_glb`.
    """
    device = pipeline.device

    if preprocess_image:
        pil_rgba = pipeline.preprocess_image(pil_rgba)
    if coords_xyz.shape[1] == 4:
        coords4 = coords_xyz.astype(np.int64)
    else:
        batch = np.zeros((coords_xyz.shape[0], 1), dtype=np.int64)
        coords4 = np.concatenate([batch, coords_xyz.astype(np.int64)], axis=1)
    coords = torch.from_numpy(coords4).to(device=device, dtype=torch.int32).contiguous()

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
    cond_512 = pipeline.get_cond([pil_rgba], 512)
    cond_1024 = (
        pipeline.get_cond([pil_rgba], 1024) if pipeline_type != "512" else None
    )

    if pipeline_type == "512":
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models["shape_slat_flow_model_512"], coords
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_512, pipeline.models["tex_slat_flow_model_512"], shape_slat
        )
        res = 512
    elif pipeline_type == "1024":
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models["shape_slat_flow_model_1024"], coords
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_1024, pipeline.models["tex_slat_flow_model_1024"], shape_slat
        )
        res = 1024
    elif pipeline_type == "1024_cascade":
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            1024,
            coords,
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_1024, pipeline.models["tex_slat_flow_model_1024"], shape_slat
        )
    elif pipeline_type == "1536_cascade":
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipeline.models["shape_slat_flow_model_512"],
            pipeline.models["shape_slat_flow_model_1024"],
            512,
            1536,
            coords,
        )
        tex_slat = pipeline.sample_tex_slat(
            cond_1024, pipeline.models["tex_slat_flow_model_1024"], shape_slat
        )
    else:
        raise ValueError(
            f"Unknown pipeline_type {pipeline_type!r}; expected one of "
            "'512', '1024', '1024_cascade', '1536_cascade'."
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipeline.decode_latent(shape_slat, tex_slat, res)


def save_mesh_glb(
    path: str | os.PathLike,
    mesh,
    texture_size: int = 4096,
    decimation_target: int = 1_000_000,
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
    ),
) -> Path:
    """Bake the textured mesh into a GLB file via ``o_voxel.postprocess.to_glb``.

    Args:
        path: output path (``.glb``).
        mesh: a ``MeshWithVoxel`` from
            :func:`inject_coords_into_trellis2` (must expose ``.vertices,
            .faces, .attrs, .coords, .layout, .voxel_size``).
        texture_size: baked texture resolution (square).  4096 is the
            public-example default.
        decimation_target: target face count for the post-bake
            decimation pass.  1M ≈ web-friendly; raise to keep more detail.
        aabb: bounding box used for GLB scaling; default is the TRELLIS
            canonical cube.

    Returns:
        ``Path`` of the written GLB.
    """
    try:
        import o_voxel  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "save_mesh_glb requires `o_voxel`.  It ships with the TRELLIS.2 "
            "setup (`bash setup.sh --o-voxel`).  Make sure your TRELLIS.2 "
            "conda env is active."
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[list(aabb[0]), list(aabb[1])],
        decimation_target=int(decimation_target),
        texture_size=int(texture_size),
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    glb.export(str(out), extension_webp=True)
    return out


__all__ = [
    "inject_coords_into_trellis2",
    "load_trellis2_pipeline",
    "save_mesh_glb",
]
