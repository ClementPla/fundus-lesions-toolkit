from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fundus_lesions_toolkit.constants import DEFAULT_COLORS


def plot_image(image, figsize=(10, 10), axis="off", title=None):
    """Plot an image"""
    plt.imshow(image)
    plt.gcf().set_size_inches(figsize)
    plt.axis(axis)
    if title:
        plt.title(title)
    plt.show()


def get_mosaic(
    batch_img: torch.Tensor,
    batch_mask: torch.Tensor = None,
    ncols=8,
    padding=2,
    alpha=0.5,
    colors=None,
):
    """Create a mosaic of images and masks"""

    from torchvision.utils import make_grid

    """Create a mosaic of images and masks"""
    if batch_img.shape[1] != 3:
        batch_img = batch_img.permute((0, 3, 1, 2))

    grid = make_grid(batch_img, nrow=ncols, padding=padding, normalize=True) * 255
    grid = grid.to(torch.uint8)

    if batch_mask is not None:
        from torchvision.utils import draw_segmentation_masks

        assert (
            batch_img.shape[0] == batch_mask.shape[0]
        ), "Batch size of images and masks must be the same"

        if batch_mask.ndim == 4 and batch_mask.shape[1] > 1:
            # Convert from probas to classes
            batch_mask = torch.argmax(batch_mask, 1, keepdim=True)

        grid_mask = make_grid(batch_mask, nrow=ncols, padding=padding)[0]
        grid_mask = F.one_hot(grid_mask, num_classes=5).permute((2, 0, 1))
        grid_mask[0] = 0  # Remove background

        grid = draw_segmentation_masks(
            grid.cpu(), grid_mask.bool().cpu(), alpha=alpha, colors=colors
        )

    return grid


def get_segmentation_mask_on_image(
    image: Union[np.ndarray, torch.Tensor],
    mask: torch.Tensor,
    alpha=0.5,
    border_alpha=0.5,
    colors=None,
    kernel_size=3,
):
    from torchvision.utils import draw_segmentation_masks
    from kornia.morphology import gradient

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = image.to(mask.device)

    if colors is None:
        colors = DEFAULT_COLORS
    if image.shape[0] != 3:
        image = image.permute((2, 0, 1))

    image = (image - image.min()) / (image.max() - image.min())
    image = (255 * image).to(torch.uint8)
    if mask.ndim == 3 and mask.shape[0] == 5:
        mask = mask.unsqueeze(0)

    if mask.ndim == 4:
        mask = torch.argmax(mask, 1)

    mask = F.one_hot(mask, num_classes=5).squeeze(0).permute((2, 0, 1))
    kernel = mask.new_ones((kernel_size, kernel_size))
    border = gradient(mask.unsqueeze(0), kernel).squeeze(0)
    border[0] = 0
    mask[0] = 0  # Remove background
    draw = draw_segmentation_masks(
        image.to(torch.uint8).cpu(),
        mask.to(torch.bool).cpu(),
        alpha=alpha,
        colors=colors,
    )

    draw = draw_segmentation_masks(
        draw, border.to(torch.bool).cpu(), alpha=1 - border_alpha, colors="white"
    )
    return draw


def plot_image_and_mask(
    image,
    mask,
    alpha=0.5,
    border_alpha=0.8,
    colors=None,
    title=None,
    figsize=(10, 10),
    labels=None,
    save_as=None,
    kernel_size=3,
):
    """Plot image and mask"""

    plt.imshow(
        get_segmentation_mask_on_image(
            image,
            mask,
            alpha,
            border_alpha=border_alpha,
            kernel_size=kernel_size,
            colors=colors,
        )
        .permute((1, 2, 0))
        .cpu()
    )
    plt.axis("off")
    if title:
        plt.title(title)
    plt.gcf().set_size_inches(figsize)
    if labels and colors:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=c, label=l) for l, c in zip(labels[1:], colors[1:])
        ]
        plt.gca().legend(handles=legend_elements, loc="upper right")

    if save_as:
        plt.savefig(save_as)
    plt.show()
