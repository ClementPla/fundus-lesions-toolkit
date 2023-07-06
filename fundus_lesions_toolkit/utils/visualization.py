from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def plot_image(image, figsize=(10, 10), axis='off', title=None):
    """Plot an image"""
    plt.imshow(image)
    plt.gcf().set_size_inches(figsize)
    plt.axis(axis)
    if title:
        plt.title(title)
    plt.show()
    


def get_mosaic(batch_img: torch.Tensor, 
               batch_mask:torch.Tensor=None,
               ncols=8, padding=2, alpha=0.5, colors=None):
    """Create a mosaic of images and masks
    """
    
    from torchvision.utils import make_grid
    """Create a mosaic of images and masks"""
    if batch_img.shape[1] != 3:
        batch_img = batch_img.permute((0,3,1,2))
    
    grid = make_grid(batch_img, nrow=ncols, padding=padding, normalize=True)*255
    grid = grid.to(torch.uint8)
    
    if batch_mask is not None:
        from torchvision.utils import draw_segmentation_masks
        
        assert batch_img.shape[0] == batch_mask.shape[0], "Batch size of images and masks must be the same"
        
        if batch_mask.ndim == 4 and batch_mask.shape[1] > 1:
            # Convert from probas to classes
            batch_mask = torch.argmax(batch_mask, 1, keepdim=True)
        
        grid_mask = make_grid(batch_mask, nrow=ncols, padding=padding)[0]
        grid_mask = F.one_hot(grid_mask, num_classes=5).permute((2,0,1))
        grid_mask[0] = 0   # Remove background

        grid = draw_segmentation_masks(grid.cpu(), grid_mask.bool().cpu(), alpha=alpha, colors=colors)    

    return grid


def get_segmentation_mask_on_image(image:Union[np.ndarray, torch.Tensor],
                                    mask:torch.Tensor, alpha=0.5, colors=None):
    from torchvision.utils import draw_segmentation_masks
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if image.shape[0] != 3:
        image = image.permute((2,0,1))
    
    image = (image - image.min())/(image.max()-image.min())
    image = (255*image).to(torch.uint8)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    if mask.shape[0]>1:
        mask = torch.argmax(mask, 0, keepdim=False)
    
    mask = F.one_hot(mask, num_classes=5).permute((2,0,1))
    mask[0] = 0   # Remove background
    
    return draw_segmentation_masks(image.cpu(), mask.bool().cpu(), alpha=alpha, colors=colors)

def plot_image_and_mask(image, mask, alpha=0.5, colors=None, title=None, figsize=(10,10), labels=None):
    """Plot image and mask"""
    plt.imshow(get_segmentation_mask_on_image(image, mask, alpha, colors).permute((1,2,0)))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.gcf().set_size_inches(figsize)
    if labels and colors:
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=l) for l,c in zip(labels[1:], colors[1:])]
        plt.gca().legend(handles=legend_elements, loc='upper right')
        
    plt.show()