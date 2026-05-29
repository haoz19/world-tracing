<div align="center">

<a href="https://www.worldlabs.ai/" target="_blank">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/worldlabs_logo_for_dark.png">
    <img src="assets/worldlabs_logo_for_light.png" alt="World Labs" width="84">
  </picture>
</a>

# World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible

### Multilayer-Geometry Diffusion (`wt`)

<a href="https://haoz19.github.io/">Hao Zhang</a><sup>1,2</sup> &nbsp;·&nbsp;
<a href="https://mbanani.github.io/">Mohamed El Banani</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://jen-haocheng.com/">Jen-Hao Cheng</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://people.csail.mit.edu/pzpzpzp1/">Paul Zhang</a><sup>1</sup> <br>
<a href="https://hawaiii.github.io/">Yi Hua</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://bmild.github.io/">Ben Mildenhall</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://christophlassner.de/">Christoph Lassner</a><sup>1</sup> &nbsp;·&nbsp;
<a href="https://vision.ai.illinois.edu/narendra-ahuja/">Narendra Ahuja</a><sup>2</sup> &nbsp;·&nbsp;
<a href="https://gengshan-y.github.io/">Gengshan Yang</a><sup>1</sup>

<sup>1</sup>World Labs &nbsp; &nbsp; <sup>2</sup>University of Illinois Urbana-Champaign

<p>
  <a href="https://arxiv.org/abs/2026.XXXXX"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/haoz19"><img alt="Hugging Face Models" src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://huggingface.co/spaces/haoz19/world-tracing-demo"><img alt="Hugging Face Demo" src="https://img.shields.io/badge/Hugging%20Face-Demo-9333EA?style=for-the-badge&logo=huggingface&logoColor=white"></a>
  <a href="https://haoz19.github.io/world-tracing-page/"><img alt="Project Website" src="https://img.shields.io/badge/Project-Website-2563EB?style=for-the-badge&logo=githubpages&logoColor=white"></a>
  <a href="./LICENSE"><img alt="License MIT" src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white"></a>
</p>

</div>

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/haoz19/world-tracing-page@assets/videos/world_tracing_demo_720p.mp4">
    <img src="assets/world_tracing_demo_poster.jpg"
         alt="Demo video — click to play (720p, ~11 MB)"
         width="100%">
  </a>
  <br>
  <sub><i>Click the image above to play the demo reel inline (720p, ~11 MB).
  Full 1440p / interactive version on the
  <a href="https://haoz19.github.io/world-tracing-page/">project page</a>.</i></sub>
</p>

---

Image-to-3D point cloud prediction via flow-matching diffusion over **layered
geometry**.  A single forward pass produces ``L`` registered XYZ maps that
together cover the visible surface *and* the (partially) occluded surfaces
behind it, giving a richer 3D scaffold than a single mono-depth map.

> The [project page](https://haoz19.github.io/world-tracing-page/) hosts
> the curated demo samples as an interactive 3D viewer.  This repository
> ships the **code** that produced those samples plus the public model
> weights so you can reproduce them on any RGBA-friendly image of your
> own.

This is the inference-only release.

> 💛 **If you find World Tracing useful**, please consider [⭐ starring this
> repo](https://github.com/haoz19/world-tracing) and [citing our
> paper](#citation) — it really helps us prioritise future releases.

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
pip install -e ".[viz,bg]"             # + BiRefNet-based foreground matting for RGB inputs
pip install -e ".[viz,flash]"          # + flash-attn (auto-detected at runtime)
pip install -e ".[viz,textured-mesh]"  # + helpers for image → textured GLB (see below)
```

The `bg` extra pulls in
[`ZhengPeng7/BiRefNet_HR`](https://huggingface.co/ZhengPeng7/BiRefNet_HR)
(MIT, SOTA dichotomous segmentation) so that `infer_rgba.py` can
auto-matte RGB inputs that don't already carry an alpha channel.
Without this extra it falls back to a fast near-white-background
heuristic and a warning.

## Quickstart

> The sample images we used for the demo video live in
> [`examples/test_images/`](examples/test_images/) -- pre-organised by
> mode (`object/`, `scene/`, `dynamic/`).  All quickstart commands below
> use the same set so you can reproduce a demo on a fresh checkout
> without finding your own inputs.
>
> Every example below runs a 4-seed sweep by default (seeds ``42, 43,
> 44, 45``) and writes one ``.rrd`` with the four samples laid out
> side-by-side along ``+X`` so you can pick the best.  Pass ``--seed N``
> to run a single deterministic seed instead.

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
python examples/infer_scene.py \
    --image  examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
    --ckpt   r69e \
    --config r69e \
    --out    /tmp/wt_scene.rrd
```

Scene mode treats the entire frame as foreground (no alpha mask) and
keeps the raw RGB (no background overwrite).  The released scene model
was trained on indoor renders without sky, so for outdoor scenes with
large sky regions you should pre-mask the sky externally (any matting
tool you like — e.g. running the same `wt.data.segment_foreground`
on the **inverted** image, or your favourite ADE20K segmenter outside
the pipeline) before feeding the result into `infer_scene.py`.

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

### 4. Choosing a different seed

The default 4-seed sweep emits all four samples in a single ``.rrd``,
spread along ``+X``.  To run a single deterministic seed, pass
``--seed``:

```bash
python examples/infer_rgba.py \
    --image  examples/test_images/object/obj063_trex_dinosaur.png \
    --ckpt   r75b \
    --seed   7 \
    --out    /tmp/wt_obj063_seed7.rrd
```

``--num-seeds K`` runs a custom sweep size starting at the default
base seed 42 (combine with ``--seed N`` to shift the base: ``--seed 100
--num-seeds 4`` runs ``100, 101, 102, 103``).  ``--num-seeds 1`` is the
fastest single-sample mode.

### 5. Textured mesh export (image → GLB)

Chains the released multilayer-geometry model with the public
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D
pipeline: we **skip** TRELLIS.2's Stage-1 sparse-structure diffusion
and feed it the voxel coords derived from our predicted XYZ.  Stages 2
+ 3 (shape SLat + texture SLat + mesh decode) then produce a textured
GLB.

```bash
# 1. one-time TRELLIS.2 setup (only needed once; ~30 min of dep install)
git clone https://github.com/microsoft/TRELLIS.2
cd TRELLIS.2
bash setup.sh --new-env --basic --flash-attn --o-voxel \
              --nvdiffrast --cumesh --flexgemm
conda activate trellis2

# 2. install wt in that same env
pip install -e /path/to/world-tracing[viz,textured-mesh]

# 3. run end-to-end (default 4-seed sweep -- writes obj014_seed{42,43,44,45}.glb)
python examples/infer_textured_mesh.py \
    --image  examples/test_images/object/obj014_leather_briefcase.png \
    --ckpt   r75b \
    --out    /tmp/wt_obj014.glb \
    --rrd    /tmp/wt_obj014.rrd \
    --trellis2-path /path/to/TRELLIS.2
```

The `--pipeline-type` flag selects the TRELLIS.2 stage configuration
(`1024_cascade` is the default — best quality / time trade-off).
By default a 4-seed sweep writes ``<out_stem>_seed{42,43,44,45}.glb`` so
you can keep the best mesh; pass ``--seed N`` (or ``--num-seeds 1``) to
run a single seed and write to the plain ``--out`` path.  ``--rrd``
additionally dumps the multilayer point cloud for sanity-check viewing
in Rerun.

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
camera.  No MoGe / pose estimator required at inference time.

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
├── model.py              ← MultilayerXYZModel (configurable wrapper around MultilayerBackbone)
├── inference.py          ← inference_diffusion / inference_diffusion_multiview / inference_video_diffusion
├── sampling.py           ← Euler ODE flow-matching sampler (replaces FMLossWrapper)
├── data.py               ← Image loaders (BiRefNet auto-matting), preprocessing, video clip
├── viz.py                ← Rerun .rrd output helpers (single image, video timeline, multi-seed)
├── intrinsics.py         ← Solve K from predicted XYZ (replaces MoGe at inference)
├── checkpoint.py         ← Released model configs + checkpoint loader + HF Hub resolver
├── postproc.py           ← Optional point-cloud cleanup (edge-flyer filter for dynamic outputs)
├── cli.py                ← Shared CLI helpers
├── textured_mesh/        ← TRELLIS.2 bridge: ours_v4 voxelisation + stage-2/3 driver
│   ├── canon.py          ← Camera ↔ TRELLIS canonical-frame transform
│   ├── voxelise.py       ← expand_cloud_ray_xyz + v4_ray_fill
│   └── pipeline.py       ← load_trellis2_pipeline + inject_coords_into_trellis2 + save_mesh_glb
└── _core/                ← Vendored deps (Wan2.1 layer init, MoGe backbone, ...)

examples/
├── infer_rgba.py         ← Single RGBA image (object model; 4-seed sweep by default)
├── infer_scene.py        ← Single scene RGB (r69e; 4-seed sweep by default)
├── infer_video.py        ← Dynamic clip (r76; 4-seed sweep by default)
└── infer_textured_mesh.py ← Image → multilayer geometry → TRELLIS.2 stages 2+3 → textured GLB (4-seed sweep by default)
```

## Hardware

Tested on a single NVIDIA A100 / H100 (80 GB) with bfloat16 autocast.

| Config | Image size | Inference time (20 steps) |
| --- | --- | --- |
| `r75b`  | 504 × 504           | ~13 s / image |
| `r69e`  | 504 × 504           | ~12 s / image |
| `r76`   | 336 × 336 × 8 frames | ~30 s / clip  |

The default 4-seed sweep is therefore ~4× the single-seed numbers above.
Smaller GPUs work with reduced ``--num-steps`` or by sampling at a
smaller resolution.

## Roadmap

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

We thank the authors for releasing their code.
