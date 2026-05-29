"""Constants used for diffusion patisserie dataset"""

RGB_KEY = "rgb"
VAE_LATENT_KEY = "vae_latent"
VAE_LATENT_PATH_KEY = "vae_latent_path"
NUM_VAE_LATENT_TOKENS_KEY = "num_vae_latent_tokens"
DEPTH_KEY = "depth"
INPUT_SHAPE_KEY = "input_shape"
BLOCK_OFFSET_KEY = "block_offset"
BLOCK_SIZE_KEY = "block_size"
ATTN_BLOCK_OFFSET_KEY = "attn_block_offset"
ATTN_BLOCK_SIZE_KEY = "attn_block_size"
NUM_FACES_KEY = "num_faces"
QUAD_L3RM_DEPTH_KEY = "quad_l3rm_depth"

CAPTION_KEY = "caption"
STYLE_KEY = "style"
T5_COMPRESSED_KEY = "t5xxl_txt_embed_compressed32"
NUM_TXT_TOKENS_KEY = "num_tokens"

# Camera related keys.
CAMERA_KEY = "camera"
POSE_EMBED_KEY = "pose_embed"
POSED_IMAGE_KEY = "posed_image"

# Video timestamp related keys.
TIMESTAMP_KEY = "timestamp"
TIMESTAMP_INTERVAL_KEY = "timestamp_interval"
VALID_FRAME_MASK_KEY = "valid_frame_mask"

# Conditional frame related keys.
COND_FRAMES_KEY = "cond_frames"
COND_FRAMES_POSE_EMBED_KEY = "cond_frames_pose_embed"
COND_FRAMES_CAMERA_KEY = "cond_frames_camera"
COND_FRAMES_TIMESTAMP_KEY = "cond_frames_timestamp"
COND_FRAMES_MASK_KEY = "cond_frames_mask"
COND_FRAMES_POSED_IMAGE_KEY = "cond_frames_posed_image"
COND_FRAMES_VAE_LATENT_KEY = "cond_frames_vae_latent"
COND_FRAMES_LOSS_MASK_KEY = "cond_frames_loss_mask"
COND_FRAMES_INPUT_SHAPE_KEY = "cond_frames_input_shape"
COND_FRAMES_QUAD_L3RM_DEPTH_KEY = "cond_frames_quad_l3rm_depth"
COND_FRAMES_IMAGE_CLIP_KEY = "cond_frames_clip_feat"

# L3RM rendering related keys.
DUMMY_L3RM_RENDERED_RGBA_KEY = "dummy_l3rm_rendered_rgba"
L3RM_RENDERED_RGBA_KEY = "l3rm_rendered_rgba"
L3RM_RENDERED_LATENTS_KEY = "l3rm_rendered_latents"
L3RM_RENDERED_LATENTS_MASK_KEY = "l3rm_rendered_latents_mask"
QUAD_L3RM_DEPTH_PATH_KEY = "quad_l3rm_depth_path"

# Rotary embeddings related keys.
X_ROTARY_EMBEDS_KEY = "x_rotary_embeds"
C_ROTARY_EMBEDS_KEY = "c_rotary_embeds"
COND_FRAMES_ROTARY_EMBEDS_KEY = "cond_frames_rotary_embeds"

# Patchify related keys.
X_PATCH_SIZE_KEY = "x_patch_size"
X_SHAPE_AFTER_PATCHIFY_KEY = "x_shape_after_patchify"
COND_FRAMES_PATCH_SIZE_KEY = "cond_frames_patch_size"
COND_FRAMES_SHAPE_AFTER_PATCHIFY_KEY = "cond_frames_shape_after_patchify"

# Used by Flux models.
T5XXL_TXT_EMBED_KEY = "t5xxl_txt_embed"
CLIP_POOLED_TXT_EMBED_KEY = "clip_pooled_txt_embed"
CFG_W_KEY = "cfg_w"

# Used by Wan models.
UMT5XXL_TXT_EMBED_KEY = "umt5xxl_txt_embed"
LOSS_MASK_KEY = "loss_mask"
STYLE_STRENGTH_KEY = "style_strength"

# TODO: Delete after legacy models are fully deprecated.
POSE_SCALE_KEY = "pose_scale"
ORIGINAL_IMAGE_SIZE_KEY = "original_image_size"
CROP_COORDS_TOP_LEFT_KEY = "crop_coords_top_left"
TARGET_IMAGE_SIZE_KEY = "target_image_size"

# Bucket batching related keys.
TARGET_HW_KEY = "target_hw"
BUCKET_ID_KEY = "bucket_id"
HEIGHT_WIDTH_FOR_COMMON_ASPECT_RATIOS = ((16, 9), (9, 16), (1, 1))
