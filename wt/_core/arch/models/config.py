"""Static constants consumed by :class:`MultilayerBackbone`.

The training-time variant of this module also defined a number of
gaussian-splatting / point-track config blobs and time-/velocity-scale
constants that the inference release never reads.  Only the four
entries below are actually consumed by the public inference path.
"""

MOGE_CONFIG_FA3 = {
    "encoder": "dinov2_vitl14",
    "remap_output": "exp",
    "output_mask": True,
    "split_head": True,
    "intermediate_layers": 4,
    "dim_upsample": [256, 128, 64],
    "dim_times_res_block_hidden": 2,
    "num_res_blocks": 2,
    "trained_area_range": [250000, 500000],
    "last_conv_channels": 32,
    "last_conv_size": 1,
    "use_fa3": True,
}

MOGE_PATCH_SIZE = 14

DIFFUSION_TIMESTEP_SCALE = 1000.0
T_MIN_CLAMP = 0.05
