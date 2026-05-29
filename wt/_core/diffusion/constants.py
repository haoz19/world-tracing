"""Conditioning-dict key constants used by the inference path.

The training-time dataset/diffusion pipeline defines many more keys
(text-embed, VAE-latent, pose-embed, multi-frame conditioning, ...), but
the released inference path only ever reads the two ``conditioning``
entries listed below from inside :class:`MultilayerBackbone` /
:class:`MultilayerXYZModel`.
"""

RGB_KEY = "rgb"
CAMERA_KEY = "camera"
