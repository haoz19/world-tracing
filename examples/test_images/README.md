# `examples/test_images/` — Demo-video sample set

This directory hosts the **exact** input samples we hand-picked for the
project's demo video (Object / Scene / Dynamic). They are the same images
used to produce the final pair-comparison clips in our paper-demo viewer.

| Category | Source |
| --- | --- |
| `object/`  | Generated RGBA assets (1376×768, RGBA PNG) |
| `scene/`   | r69e training-set scenes (504×504, RGBA PNG, alpha=255) |
| `dynamic/` | 16-frame clips evenly subsampled from DAVIS / Consistent4D / human-dynamic |

Each category mixes a small "tier-1" set of visually striking examples
with additional curated samples that round out the variety of object
types, scene styles, and dynamic motion.

## Layout

```
test_images/
├── MANIFEST.json         # canonical mapping (stem, label, tier, source paths)
├── object/<stem>.png     # 1376×768 RGBA, ready for examples/infer_rgba.py
├── scene/<stem>.png      # 504×504 RGBA (alpha=255), ready for examples/infer_scene.py
└── dynamic/<stem>/       # 16 PNG frames, ready for examples/infer_video.py
    ├── 00000.png
    ├── 00001.png
    └── ...
```

`<stem>` for `dynamic/` includes the dataset prefix (`davis__`,
`consistent4d_in_the_wild__`, `consistent4d_synthetic__`, `human__`) to
avoid name collisions between datasets that share the same clip name.

## How to use

From the repo root, with the checkpoint downloaded according to
[`checkpoints/README.md`](../../checkpoints/README.md):

```bash
# Object (r75b)
python examples/infer_rgba.py \
    --image examples/test_images/object/obj014_leather_briefcase.png \
    --ckpt  /path/to/r75b_spikeskip_hardaug.pt \
    --out   /tmp/obj014.rrd

# Scene (r69e); for outdoor scenes with large sky, pre-mask the sky externally
python examples/infer_scene.py \
    --image examples/test_images/scene/scene_outdoor_14_brooklyn_apartment__seed61.png \
    --ckpt  /path/to/r69e_v2_evermotion_ithappy_504.pt \
    --out   /tmp/scene_brooklyn.rrd

# Dynamic (r76)
python examples/infer_video.py \
    --frames_dir examples/test_images/dynamic/davis__camel \
    --ckpt       /path/to/r76_video_dynamic_16frame.pt \
    --out        /tmp/dyn_camel.rrd
```

Tip: append `--layer-timeline` to `infer_rgba.py` / `infer_scene.py` for
the cumulative-layer scrubbing visualisation.

## Reproducing the sample set

The samples were collected from the upstream demo-video pipeline using:

```bash
python _collect_demo_samples.py             # writes test_images/{object,scene,dynamic}/
python _collect_demo_samples.py --dry-run   # report only, do not copy
```

Sources (internal paths, recorded in `MANIFEST.json` for reference):

* `object/`  ← `auto-experiments/video_static/results/generated_object/<stem>/input/<stem>.png`
* `scene/`   ← `rgba` field of `auto-experiments/video_results_scene/real/npz/<stem>.npz`
* `dynamic/` ← every-Kth frame of:
  * DAVIS: `test_images/Video-DAVIS-RGBA/<clip>/frames/*.png`
  * Consistent4D: `test_images/Consistent4D/{in-the-wild,synthetic}/<name>/*.png`
  * Human: `test_images/Video-human-dynamic/<id>/frames/*.png`

The selection JSONs that drive the script live in
`auto-experiments/video-finial-selection/{object,scene,dynamic}_pair_selection.json`.
