import einops
import structlog
import torch
from jaxtyping import Float
from wt._core.vendor.moge.moge2_model import MoGeModel
from torch import Tensor

logger = structlog.get_logger(__name__)


class MoGe2Runner:
    """
    Class to run depth inference using the MoGe2 model.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = MoGeModel.from_pretrained(model_name="moge-2-vitl-normal").to(
            self.device
        )

    @torch.no_grad()
    def __call__(
        self, imgs: Float[Tensor, "B H W 3"], chunk_size: int = 32
    ) -> dict[str, Float[Tensor, "..."]]:
        """
        Run moge2 inference on a batch of images.

        Args:
            imgs: The input images in 0-1 range.
            chunk_size: The size of the chunks to process the images in.

        Returns:
            `output_dict` has keys "points", "depth", "mask", "mask_confidence",
            "normal" (optional) and "intrinsics",
            The maps are in the same size as the input image.
            {
                "points": (B, H, W, 3),    # point map in OpenCV camera coordinate
                    system (x right, y down, z forward). For MoGe-2, the point map
                     is in metric scale.
                "depth": (B, H, W),        # depth map
                "normal": (B, H, W, 3)     # normal map in OpenCV camera coordinate.
                "mask": (B, H, W),         # a binary mask for valid pixels.
                "intrinsics": (B, 3, 3),   # normalized camera intrinsics
                "mask_confidence": (B, H, W),  # soft mask confidence in [0, 1]
            }
        """
        assert (
            imgs.ndim == 4
        ), f"Input images must be of shape (B, H, W, 3), but got shape {imgs.shape}"
        assert imgs.shape[-1] == 3, "Input images must have channel 3."
        assert imgs.max() <= 1.0 and imgs.min() >= 0.0, "Input must be in [0, 1]."
        imgs = einops.rearrange(imgs, "B H W C -> B C H W")
        bs, _, height, width = imgs.shape
        # preallocate output tensors
        # normal is always available
        output_dict = {
            "points": torch.empty(bs, height, width, 3, device=self.device),
            "depth": torch.empty(bs, height, width, device=self.device),
            "mask": torch.empty(bs, height, width, device=self.device),
            "mask_confidence": torch.empty(bs, height, width, device=self.device),
            "normal": torch.empty(bs, height, width, 3, device=self.device),
            "intrinsics": torch.empty(bs, 3, 3, device=self.device),
        }
        for idx in range(0, bs, chunk_size):
            chunk = imgs[idx : idx + chunk_size]
            output = self.model.infer(chunk)
            assert (
                "points" in output
                and "depth" in output
                and "mask" in output
                and "mask_confidence" in output
                and "normal" in output
                and "intrinsics" in output
            ), f"Output must have key 'points', but got {output.keys()}"
            for k, v in output.items():
                output_dict[k][idx : idx + chunk_size] = v
        return output_dict
