# World Tracing (`wt`) — Multilayer-Depth Diffusion

Image-to-3D point cloud prediction via flow-matching diffusion over **layered
depth**.  A single forward pass produces ``L`` registered depth maps that
together cover the visible surface *and* the (partially) occluded surfaces
behind it, giving a richer 3D scaffold than a single mono-depth map.

> **Project page (live demos):**
> [https://haoz19.github.io/world-tracing-page/](https://haoz19.github.io/world-tracing-page/)
>
> The project page hosts the same 37 curated samples (10 objects + 8 scenes
> + 19 dynamic clips) as an interactive 3D viewer.  This repository ships
> the **code** that produced those samples plus the public model weights so
> you can reproduce them on any RGBA-friendly image of your own.

This is the inference-only release.  Training code, dataset preparation,
and evaluation utilities will follow.

## Released checkpoints

All three checkpoints are hosted on **Hugging Face Hub**:

| config name | task | image size | params | Hugging Face repo |
| --- | --- | --- | --- | --- |
| `r75b` | object | 504 × 504 | 1.7 B | [`haoz19/object-model-6layer`](https://huggingface.co/haoz19/object-model-6layer) |
| `r69e` | scene | 504 × 504 | 1.5 B | [`haoz19/scene-model-6layer`](https://huggingface.co/haoz19/scene-model-6layer) |
| `r76`  | dynamic object (16 frames) | 336 × 336 | 2.1 B | [`haoz19/dynamic-model-16frame`](https://huggingface.co/haoz19/dynamic-model-16frame) |

Pass `--ckpt <config-name>` (e.g. `--ckpt r75b`) and `wt` will fetch the
weights from the Hub on first use and cache them under
`~/.cache/huggingface/`.  You can also pass an `hf://` URI or a local
`.pt` path -- see [the checkpoint section](#checkpoint-handling) below.

## Installation

```bash
git clone https://github.com/haoz19/world-tracing.git
cd world-tracing
pip install -e ".[viz]"
```

Tested with Python ≥ 3.10 on Linux with CUDA 12.  The base install pulls
in `torch`, `numpy`, `Pillow`, `opencv-python`, `einops`, `safetensors`,
`huggingface_hub`, `structlog`, `beartype`, `jaxtyping`.  The `viz` extra
adds [`rerun-sdk`](https://rerun.io/) (for the ``.rrd`` viewer) and
`scipy`.

Optional extras:

```bash
pip install -e ".[viz,sky]"     # + transformers, for sky segmentation
pip install -e ".[viz,flash]"   # + flash-attn (auto-detected at runtime)
```

## Quickstart

> The 30 + 20 + 41 sample images we used for the demo video live in
> [`examples/test_images/`](examples/test_images/) -- 30 objects, 20
> scenes, and 41 dynamic clips (16 frames each).  All quickstart commands
> below use the same set so you can reproduce a demo on a fresh checkout
> without finding your own inputs.

### 1. Single RGBA / RGB object image (`r75b`)

```bash
python examples/infer_rgba.py \
    --image  examples/test_images/object/obj014_leather_briefcase.png \
    --ckpt   r75b \
    --config r75b \
    --out    /tmp/wt_obj014.rrd

rerun /tmp/wt_obj014.rrd
```

If your input is an RGB image whose object is matted onto a near-white
background (common for SAM / Stable-Diffusion outputs), `wt` auto-derives
a binary alpha; pass `--no-auto-alpha` to disable that.

> Tip: pass ``--layer-timeline`` to log the prediction along a ``layer``
> timeline.  Scrubbing the timeline slider in Rerun adds one layer at a
> time and makes it obvious how each layer carves out the occluded
> geometry behind the previous one.

### 2. Scene RGB (`r69e`)

```bash
pip install -e ".[viz,sky]"   # extra dependency: transformers (for sky segmentation)

python examples/infer_scene.py \
    --image  examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
    --ckpt   r69e \
    --config r69e \
    --out    /tmp/wt_scene.rrd
```

Scene mode treats the entire frame as foreground (no alpha mask) and
keeps the raw RGB (no background overwrite).  The scene model was
trained on indoor renders without sky, so for outdoor scenes
`infer_scene.py` automatically runs an [ADE20K
SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
to mark sky pixels as **invalid** before inference; without this, sky
pixels are pushed to a wildly large depth and dominate the layer-0
output.

Useful flags:

```bash
--no-sky-segment             # disable auto sky segmentation
--sky-mask path/to/sky.png   # supply your own white=sky mask
--save-sky-mask /tmp/sky.png # dump the predicted sky mask for inspection
--sky-model      ...         # swap in a different ADE20K segmenter
--layer-timeline             # log layers along a Rerun timeline (scrub layer-by-layer)
```

### 3. Dynamic clip (`r76`)

```bash
python examples/infer_video.py \
    --image_dir examples/test_images/dynamic/davis__camel/   # 16 PNG frames
    --ckpt      r76 \
    --config    r76 \
    --out       /tmp/wt_camel.rrd
```

The resulting `.rrd` uses the `frame` timeline; scrub the slider in
Rerun to animate the predicted point cloud over time.  All frames share
a single crop so the temporal-attention blocks can establish per-pixel
correspondences.  Pass ``--frame_indices "0,2,4,6,8,10,12,14"`` to pick
a subset of the 16 supplied frames.

### 4. Multi-seed sampling

```bash
python examples/infer_multiseed.py \
    --image      examples/test_images/object/obj063_trex_dinosaur.png \
    --num_seeds  4 \
    --ckpt       r75b \
    --config     r75b
```

Four independent denoising trajectories of the same image, laid out
side-by-side along ``+X`` so you can compare the variation in occluded
layers.

## Checkpoint handling

`--ckpt` accepts any of:

1. **Bare config name** (`--ckpt r75b`) — fetched from the default Hugging
   Face repo for that config (see the table at the top).
2. **HF shorthand** (`--ckpt hf://haoz19/object-model-6layer`) — uses
   `model.pt` from the given repo.  Add a file path for a non-default
   filename: `--ckpt hf://my-fork/object/model.pt`.
3. **Local path** (`--ckpt /path/to/checkpoint.pt`) — useful for
   fine-tuned weights.  Accepts both raw `state_dict` and
   `{"model_state_dict": ..., "ema_state_dict": ...}` formats; EMA is
   preferred when present.

Resolution and download happen in `wt.checkpoint.resolve_ckpt_path` --
the cached file lives under `~/.cache/huggingface/hub/` so subsequent
runs are instant.

## What you get back

The released models predict **per-layer geometry only**:

| name | shape | meaning |
| --- | --- | --- |
| `xyz_pred` | ``[B, L, H, W, 3]`` | Per-layer XYZ in camera space (metric units for `r75b` / `r76`; relative scale for `r69e` median-log) |

The per-layer validity mask is taken from the input alpha (the model's
output is unmasked geometry over the full grid); per-pixel colour is
sampled from the input RGB at the corresponding location.  No colour
or visibility is predicted by the model.

Camera intrinsics for the predicted point cloud can be recovered from
layer-0 with [`wt.solve_intrinsics_from_xyz`](wt/intrinsics.py); this lets
you turn the prediction into a textured mesh or render it through any
camera.  No MoGe / VGGT / pose estimator required at inference time.

## Background handling

The model's frozen image encoder reads the raw RGB pixels regardless of
the validity mask.  If your input has a coloured background, the encoder
will treat it as valid content and the model will produce "ghost"
geometry over it.

`preprocess_rgba_for_model` therefore overwrites the background (alpha ≤
127) region with a fixed RGB triple before the resize.  The default is
black ``(0, 0, 0)``, which matches the training-set renders (Objaverse +
composite scenes) and the `bg_randomize` augmentation that ran during
training.  Pass ``--bg-color none`` to keep the raw RGB (only useful for
scene mode or explicit ablations).

## Package layout

```
wt/                       ← installable Python package
├── model.py              ← MultilayerXYZModel (configurable wrapper around ThreersV2)
├── inference.py          ← inference_diffusion / inference_diffusion_multiview / inference_video_diffusion
├── sampling.py           ← Euler ODE flow-matching sampler (replaces FMLossWrapper)
├── data.py               ← Image loaders, alpha-aware crop+resize, video clip preprocess
├── viz.py                ← Rerun .rrd output helpers (single image, video timeline, multi-seed)
├── intrinsics.py         ← Solve K from predicted XYZ (replaces MoGe at inference)
├── checkpoint.py         ← Released model configs + checkpoint loader + HF Hub resolver
├── postproc.py           ← Optional point-cloud cleanup (edge-flyer filter for dynamic outputs)
├── cli.py                ← Shared CLI helpers
└── _internal/            ← Vendored deps (Wan2.1 layer init, MoGe backbone, VGGT layer scale, ...)

examples/
├── infer_rgba.py         ← Single RGBA image (object model)
├── infer_scene.py        ← Single scene RGB (r69e)
├── infer_video.py        ← Dynamic clip (r76)
└── infer_multiseed.py    ← N seeds on one image
```

## Hardware

Tested on a single NVIDIA A100 / H100 (80 GB) with bfloat16 autocast.

| Config | Image size | Inference time (20 steps) |
| --- | --- | --- |
| `r75b`  | 504 × 504           | ~13 s / image |
| `r69e`  | 504 × 504           | ~12 s / image |
| `r76`   | 336 × 336 × 8 frames | ~30 s / clip  |

Multi-seed is N× longer (it runs N independent samplings).  Smaller GPUs
work with reduced ``--num-steps`` or by sampling at a smaller resolution.

## Roadmap

* **Textured-mesh export.**  An end-to-end "image → multilayer depth →
  voxelisation (`v4_ray_fill`) → [TRELLIS.2](https://github.com/microsoft/TRELLIS)
  stage 2 + 3 → GLB" pipeline that produces a clean textured mesh from a
  single image.  Will be added as `examples/infer_textured_mesh.py` once
  the public TRELLIS.2 integration is stabilised.
* **Training code.**  Currently only inference is open-sourced.  Training
  scripts and dataset preparation tooling will follow.
* **More published checkpoints.**  Updated `r75b` / `r69e` / `r76` from
  later training rounds, and a single-image multi-view variant.

## Citation

```bibtex
@misc{zhang2026worldtracing,
  title         = {World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible},
  author        = {Hao Zhang and Mohamed El Banani and Jen-Hao Cheng and Paul Zhang
                   and Yi Hua and Ben Mildenhall and Christoph Lassner
                   and Narendra Ahuja and Gengshan Yang},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

MIT — see ``LICENSE``.

## Acknowledgements

The model architecture borrows from:

* [MoGe](https://huggingface.co/microsoft/moge-2-vitl) (DINOv2 encoder backbone)
* [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (timestep embedding + initialisation)
* [VGGT](https://github.com/facebookresearch/vggt) (LayerScale)

We thank the authors for releasing their code.
