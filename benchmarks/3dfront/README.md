# 3D-FRONT benchmark (Table 2)

Evaluation code and sample list for the 3D-FRONT scene benchmark in Table 2
of the paper: 50 frames from 10 3D-FRONT rooms that were held out from
training.  The ground truth (input RGB, first 6 depth-peeled layers,
per-layer validity masks, intrinsics; 272 MB) is hosted next to the scene
model in [`haoz19/scene-model-6layer`](https://huggingface.co/haoz19/scene-model-6layer)
under `eval/3dfront_test50/` and is downloaded on first use.

## Reproduce WT-S

```bash
pip install -e ".[eval]"
huggingface-cli login            # the HF repo is behind an instant-approval access form
python benchmarks/3dfront/eval_3dfront.py
```

This runs the released `r69e` checkpoint on all 50 frames with 8 diffusion
seeds (about 40 min on one H100; `--num-seeds 1` for a quick check) and
prints every seed plus the mean and std over seeds.  Expected output:

| | MAE | RMSE | AbsRel | L0 CD-L1 | L0 CD-L2 | L0 F@0.05 | All CD-L1 | All CD-L2 | All F@0.05 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper (WT-S, 1 seed) | 0.0102 | 0.0215 | 0.0114 | 0.0093 | 0.000204 | 0.9867 | 0.0216 | 0.001255 | 0.8951 |
| Released `r69e`, mean of 8 seeds | 0.0085 | 0.0194 | 0.0094 | 0.0081 | 0.000146 | 0.9925 | 0.0208 | 0.001240 | 0.8998 |
| Released `r69e`, std over 8 seeds | 0.0004 | 0.0004 | 0.0004 | 0.0003 | 0.000013 | 0.0012 | 0.0005 | 0.000153 | 0.0056 |

The paper row was computed with an earlier checkpoint of the same training
run (about 100k iterations before the released one) and a single seed.

## Evaluate your own method

Write one `<stem>.npz` per frame, with the stems listed in
[`samples_3dfront_test50.json`](samples_3dfront_test50.json) (e.g.
`00_21a0379d_MasterBedroom-142318_f03`), containing

* `depth`: `[L, 504, 504]` (or `[504, 504]`) camera-space z of layer 0..L-1,
  in any scale (a scale and shift are fitted per frame);
* `mask` (optional): same shape, bool; defaults to `isfinite(depth)`.

Then

```bash
python benchmarks/3dfront/eval_3dfront.py --pred-dir my_preds/
```

The input to your method is the `rgb` array of each ground-truth `.npz`
(`[3, 504, 504]`, float32 in [0, 1]); the camera is the `intrinsics` array.
Single-layer predictions get the L0 metrics only.

## Protocol

* **Frames.** 512 × 512 renders (field of view 70–108°, centred principal
  point) resized to 504 × 504.  The held-out rooms and frames are listed in
  `samples_3dfront_test50.json`.
* **Ground truth.** The first 6 depth-peeled surfaces of every pixel, divided
  by the per-frame median depth, so all errors and distances are relative to
  the median scene depth rather than metres.  Layers that a pixel does not
  have are marked invalid.
* **Our model.** Input RGB; the L0 valid-depth mask broadcast to all 6
  layers as `gt_mask`; 20 Euler steps; seed `42 + i + 1000 k` for frame `i`
  and seed index `k`.  Only the predicted z is scored, unprojected with the
  ground-truth intrinsics.
* **L0 depth (MAE / RMSE / AbsRel).** Per-frame least-squares scale and
  shift of predicted L0 depth onto GT over valid pixels; per-frame metrics
  are averaged with pixel-count weights.
* **L0 CD.** The same L0 fit, unprojected with the GT intrinsics.
* **All-layer CD.** One scale and shift fitted jointly over all layers;
  layer `l` is kept only where GT layer `l` is valid.
* **Chamfer.** CD-L1 is the mean of the two directed mean nearest-neighbour
  distances, CD-L2 the same with squared distances, F@0.05 the F-score at
  threshold 0.05.  Each cloud is randomly subsampled to at most 50k points
  (`RandomState(12345)`); metrics are averaged over frames.
* **Chamfer floor.** GT points come from the renderer's point maps (resized
  to 504), predictions are unprojected with the intrinsics, so even the GT
  depth itself scores L0 CD-L1 0.0028 / All CD-L1 0.0031 (F@0.05 = 1.0).
  This is the same for every method and was the case for all numbers in the
  paper.

The frames are renders of [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
scenes, provided for evaluation only and subject to the 3D-FRONT terms of
use.
