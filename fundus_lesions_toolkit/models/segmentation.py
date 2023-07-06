import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
import os
from typing import Literal, Union
from .checkpoints import DOWNLOADABLE_MODELS, MODELS_TRAINED_WITH_DROPOUT
from ..utils.images import autofit_fundus_resolution, reverse_autofit_tensor
import warnings

Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]
TrainedOn = Literal["ALL"]


_last_model = (None, None)


def segment(image:np.ndarray, arch: Architecture='unet', 
            encoder: EncoderModel='timm-resnest50d',
            weights:TrainedOn='All',
            image_resolution = 1536,
            autofit_resolution = True,
            device: torch.device="cuda"):
    """Segment fundus image into 5 classes: background, CTW, EX, HE, MA

    Args:
        image (np.ndarray):   Fundus image of size HxWx3
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'timm-resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        image_resolution (int, optional): Defaults to 1536.
        autofit_resolution (bool, optional):  Defaults to True.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size 5xHxW)
    """
    model = get_model(arch, encoder, weights, device)
    if autofit_resolution:
        image, transforms = autofit_fundus_resolution(image, image_resolution)
    image = (image/255.).astype(np.float32)
    model.eval()
    tensor = torch.from_numpy(image).permute((2,0,1)).unsqueeze(0).to(device)
    tensor = Ftv.normalize(tensor, mean = [0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225))
    with torch.inference_mode():
        pred = F.softmax(model(tensor), 1)
        
    pred = pred.squeeze(0)
    if autofit_resolution:
        pred = reverse_autofit_tensor(pred, **transforms)
        all_zeros = ~torch.any(pred, dim=0) # Find all zeros probabilities 
        pred[0, all_zeros] = 1 # Assign them to background
    
    return pred


def batch_segment(batch:Union[torch.Tensor, np.ndarray], arch: Architecture='unet', encoder: EncoderModel='timm-resnest50d', 
                  weights:TrainedOn='All', 
                  already_normalized = False,
                  device: torch.device="cuda"):
    """Segment batch of fundus images into 5 classes: background, CTW, EX, HE, MA

    Args:
        batch (Union[torch.Tensor, np.ndarray]): Batch of fundus images of size BxHxWx3 or Bx3xHxW
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'timm-resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        already_normalized (bool, optional): Defaults to False.
        device (torch.device, optional):  Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size Bx5xHxW)
    """
    
    model = get_model(arch, encoder, weights, device)
    model.eval()
    
    # Check if batch is torch.Tensor or np.ndarray. If np.ndarray, convert to torch.Tensor
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch) # Convert to torch.Tensor   
    
    batch = batch.to(device)
    
    # Check if dimensions are BxCxHxW. If not, transpose
    if batch.shape[1] != 3:
        batch = batch.permute((0,3,1,2))
        
    
    # Check if batch is normalized. If not, normalize it
    if not already_normalized:
        batch = batch/255.
        batch = Ftv.normalize(batch, mean = [0.485, 0.456, 0.406], std=(0.229, 0.224, 0.225))
    
    with torch.inference_mode():
        pred = F.softmax(model(batch), 1)
    
    return pred

    
    
def get_model(arch:Architecture='unet', encoder:EncoderModel='timm-resnest50d', weights:TrainedOn='All', device: torch.device="cuda"):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'timm-resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        nn.Module: Torch segmentation model
    """
    
    global _last_model
    if _last_model[0] == (arch, encoder, weights):
        model = _last_model[1]
    else:
        model = segmentation_model(arch=arch, encoder=encoder, weights=weights).to(device=device)
        _last_model = ((arch, encoder, weights), model)
    return model
    
    
def segmentation_model(arch:Architecture, encoder:EncoderModel, weights:TrainedOn):
    arch = arch.lower()
    weights = weights.lower()
    encoder = encoder.lower()
    model_key = (arch, encoder, weights)
    assert model_key in DOWNLOADABLE_MODELS.keys(), f"Wrong combinations of architecture, encoder and weights asked {(arch, encoder, weights)}. Call list_models() to see all configurations acceptable"

    import segmentation_models_pytorch as smp
    model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None, in_channels=3, classes=5)
    
    url = DOWNLOADABLE_MODELS[model_key]
    model_name = f'{arch}_{encoder}_{weights}.ckpt'
    model_dir = os.path.join(torch.hub.get_dir(), 'checkpoints/fundus_segmentation_toolkit/segmentation')
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', file_name=model_name, model_dir=model_dir)
    
    model.load_state_dict(state_dict=state_dict, strict=True)
    if model_key in MODELS_TRAINED_WITH_DROPOUT:
        model = set_dropout(model, initial_value=0.2)
    return model

def set_dropout(model, initial_value=0.0):
    warnings.warn(f"Setting dropout to {initial_value}")
    for k, v in list(model.named_modules()):
        if 'drop' in k.split('.'):
            parent_model = model
            for model_name in k.split('.')[:-1]:
                parent_model = getattr(parent_model, model_name)
            setattr(parent_model, 'drop', nn.Dropout2d(p=initial_value))

    return model