"""Textured-mesh export via the public TRELLIS.2 stack.

This module bridges our multilayer-geometry point cloud and the public
``microsoft/TRELLIS.2-4B`` image-to-3D pipeline.  We **skip** TRELLIS.2's
Stage-1 sparse-structure diffusion and feed it the voxel coords derived
from our predicted XYZ instead, then run TRELLIS.2's Stage 2 (shape SLat)
and Stage 3 (texture SLat + decode) to produce a textured GLB.

The high-level workflow is:

1. Run ``wt.inference_diffusion`` to get ``(L, H, W, 3)`` XYZ + masks for
   an RGBA image (this is the standard ``examples/infer_rgba.py`` path).
2. Densify between adjacent depth layers via :func:`expand_cloud_ray_xyz`
   so the voxel grid is not a paper-thin shell.
3. Centre + isotropic-scale the cloud into the TRELLIS canonical cube
   ``[-0.5, 0.5]^3`` via :class:`CanonicalTransform`.
4. Voxelise + morphologically close + fill holes via :func:`v4_ray_fill`.
5. Inject the resulting ``(M, 4) int`` coords into
   ``Trellis2ImageTo3DPipeline.sample_shape_slat`` (or ``..._cascade``)
   followed by ``sample_tex_slat`` and ``decode_latent``.

See :mod:`wt.textured_mesh.pipeline` for the orchestration entry point
and ``examples/infer_textured_mesh.py`` for a self-contained CLI.

Hard dependency: a working clone of the public TRELLIS.2 repo
(https://github.com/microsoft/TRELLIS.2) with ``setup.sh`` already run
(installs ``trellis2``, ``o_voxel``, ``flex_gemm``, ``flash-attn``, and
either ``cumesh`` or ``nvdiffrast``).  Point ``--trellis2-path`` at that
checkout when invoking the example script.
"""

from __future__ import annotations

from wt.textured_mesh.canon import (
    CanonicalTransform,
    aggregate_camera_cloud,
    canon_inverse,
    compute_canonical_transform,
)
from wt.textured_mesh.pipeline import (
    inject_coords_into_trellis2,
    load_trellis2_pipeline,
    save_mesh_glb,
)
from wt.textured_mesh.voxelise import (
    expand_cloud_ray_xyz,
    v4_ray_fill,
)

__all__ = [
    "CanonicalTransform",
    "aggregate_camera_cloud",
    "canon_inverse",
    "compute_canonical_transform",
    "expand_cloud_ray_xyz",
    "inject_coords_into_trellis2",
    "load_trellis2_pipeline",
    "save_mesh_glb",
    "v4_ray_fill",
]
