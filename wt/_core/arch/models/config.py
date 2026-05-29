MOGE_CONFIG = {
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
    "use_fa3": False,
}

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


POINT_TRACK_KEYS = ["point_track"]
POINT_TRACK_CONF_KEYS = ["point_track_vis", "point_track_conf"]

VOXEL_UNET_CONFIG = {
    "c_in": 14,  # gaussians, mean (3), quat (4), scale (3), opacity (1), color (3)
    "c_out": 14,
    "zero_init": False,
    "chunk_mode": False,
}

# gaussian properties to average over / ignore
PROPERTY_LIST_AVG = [
    "mean",
    "scale",
    "quaternion",
    "opacity",
    "feature",
    "trajectory",
    "time",
    "duration",
    "velocity",
    "angular_velocity",
]

PROPERTY_LIST_INVALID = [
    "cov",
]

FTGS_TIME_SCALE = 0.01
FTGS_DURATION_SCALE = 10.0
FTGS_VELOCITY_SCALE = 0.1
FTGS_ANGULAR_VELOCITY_SCALE = 0.1

DIFFUSION_TIMESTEP_SCALE = 1000.0
T_MIN_CLAMP = 0.05
