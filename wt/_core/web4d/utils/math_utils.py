import torch
from jaxtyping import Bool
from torch import Tensor


def greedy_max_coverage(
    masks: list[Bool[Tensor, "N"]], num_select: int
) -> tuple[list[int], Bool[Tensor, "N"]]:
    """
    Greedy approximation for selecting K masks that maximize the union coverage.

    Args:
        masks: a list of boolean tensors of shape (N,).
        num_select: Number of masks to select.

    Returns:
        selected: Indices of the chosen masks.
        covered: a mask of shape (N,) for the union of the chosen masks.
    """
    assert len(masks) > 0, "No masks to select from"
    num_masks = len(masks)
    num_pts = masks[0].shape[0]
    covered = torch.zeros(num_pts, dtype=torch.bool, device=masks[0].device)
    selected = []
    remaining = set(range(num_masks))

    for _ in range(num_select):
        best_gain = -1
        best_idx = None
        # Compute gains
        for i in remaining:
            gain = int(((masks[i] & ~covered).sum()).item())
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain <= 0:
            break
        selected.append(best_idx)
        covered |= masks[best_idx]
        remaining.remove(best_idx)

    return selected, covered


def random_sample_points_box(
    height: int, width: int, num_pts: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Randomly sample integer points in a box of size height x width
    """
    rand_x = torch.randint(0, width, (num_pts,), device=device)
    rand_y = torch.randint(0, height, (num_pts,), device=device)
    new_xy = torch.cat(
        [
            torch.zeros_like(rand_x[:, None]),
            rand_x[:, None],
            rand_y[:, None],
        ],
        dim=-1,
    )
    return new_xy


def random_sample_points_mask(
    mask: torch.Tensor, num_pts: int, device: str = "cpu"
) -> torch.Tensor:
    """
    Randomly sample integer points in a 2D mask and sort them by y-coordinate
    Args:
        mask: Boolean tensor of shape (H, W)
        num_pts: Number of points to sample
        device: Device to place tensors on
    Returns:
        torch.Tensor: Points of shape (num_pts, 2) containing (x,y) coordinates
    """
    mask = mask.to(device)
    _, width = mask.shape
    mask = mask.view(-1).float()  # Convert to float for multinomial
    num_pts = min(num_pts, mask.sum().item())
    sampled_indices = torch.multinomial(mask, num_pts, replacement=True)
    sampled_points = torch.zeros((num_pts, 2), device=device)
    sampled_points[:, 0] = sampled_indices % width  # x coordinates
    sampled_points[:, 1] = sampled_indices // width  # y coordinates

    # Sort points based on y-coordinate
    sorted_indices = torch.argsort(sampled_points[:, 1])
    sampled_points = sampled_points[sorted_indices]

    return sampled_points
