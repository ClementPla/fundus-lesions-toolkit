import warnings
from typing import Literal, Union, List, Tuple
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
from fundus_lesions_toolkit.models.hf_hub import download_model
from fundus_lesions_toolkit.constants import Dataset
from fundus_data_toolkit.functional import (
    autofit_fundus_resolution,
    reverse_autofit_tensor,
)
from fundus_data_toolkit.config import get_normalization


Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]


def segment(
    image: np.ndarray,
    arch: Architecture = "unet",
    encoder: EncoderModel = "seresnext50_32x4d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    image_resolution=1024,
    autofit_resolution=True,
    reverse_autofit=True,
    mean=None,
    std=None,
    return_features=False,
    return_decoder_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Segment fundus image into 5 classes: background, CTW, EX, HE, MA

    Args:
        image (np.ndarray):   Fundus image of size HxWx3
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        image_resolution (int, optional): Defaults to 1024.
        mean (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_MEAN.
        std (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_STD.
        autofit_resolution (bool, optional):  Defaults to True.
        return_features (bool, optional): Defaults to False. If True, returns also the features map of the i-th encoder layer. See features_layer.
        features_layer (int, optional): Defaults to 3. If return_features is True, returns the features map of the i-th encoder layer.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size 5xHxW)
    """
    model = get_model(arch, encoder, train_datasets, device, compile=compile)
    model.eval()
    h, w, c = image.shape
    if autofit_resolution:
        image, roi, transforms = autofit_fundus_resolution(
            image, image_resolution, return_roi=True
        )

    image = (image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]
    tensor = Ftv.normalize(tensor, mean=mean, std=std)

    with torch.inference_mode():
        features = model.encoder(tensor)
        pre_segmentation_features = model.decoder(*features)
        pred = model.segmentation_head(pre_segmentation_features)
        pred = F.softmax(pred, 1)
        if return_features or return_decoder_features:
            assert not reverse_autofit, "reverse_autofit is not compatible with return_features or return_decoder_features"
            out = [pred]
            if return_features:
                out.append(features[features_layer])
            if return_decoder_features:
                out.append(pre_segmentation_features)
            return tuple(out)

    pred = pred.squeeze(0)
    if reverse_autofit and autofit_resolution:
        pred = reverse_autofit_tensor(pred, **transforms)
        all_zeros = ~torch.any(pred, dim=0)  # Find all zeros probabilities
        pred[0, all_zeros] = 1  # Assign them to background
    return pred


def batch_segment(
    batch: Union[torch.Tensor, np.ndarray],
    arch: Architecture = "unet",
    encoder: EncoderModel = "seresnext50_32x4d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    already_normalized=False,
    mean=None,
    std=None,
    return_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Segment batch of fundus images into 5 classes: background, CTW, EX, HE, MA

    Args:
        batch (Union[torch.Tensor, np.ndarray]): Batch of fundus images of size BxHxWx3 or Bx3xHxW
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        already_normalized (bool, optional): Defaults to False.
        mean (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_MEAN.
        std (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_STD.
        return_features (bool, optional): Defaults to False. If True, returns also the features map of the i-th encoder layer. See features_layer.
        features_layer (int, optional): Defaults to 3. If return_features is True, returns the features map of the i-th encoder layer.
        device (torch.device, optional):  Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size Bx5xHxW)
    """

    model = get_model(arch, encoder, train_datasets, device, compile=compile)
    model.eval()

    # Check if batch is torch.Tensor or np.ndarray. If np.ndarray, convert to torch.Tensor
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)  # Convert to torch.Tensor

    batch = batch.to(device)

    # Check if dimensions are BxCxHxW. If not, transpose
    if batch.shape[1] != 3:
        batch = batch.permute((0, 3, 1, 2))

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]

    # Check if batch is normalized. If not, normalize it
    if not already_normalized:
        batch = batch / 255.0
        batch = Ftv.normalize(batch, mean=mean, std=std)

    with torch.inference_mode():
        pred = F.softmax(model(batch), 1)

    if return_features:
        features = model.encoder(batch)
        pred = model.segmentation_head(model.decoder(features))
        return F.softmax(pred, 1), features[features_layer]

    return pred


@lru_cache(maxsize=2)
def get_model(
    arch: Architecture = "unet",
    encoder: EncoderModel = "resnest50d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    device: torch.device = "cuda",
    compile: bool = False,
):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        nn.Module: Torch segmentation model
    """
    model = download_model(arch, encoder, train_datasets).to(device=device)
    set_dropout(model, initial_value=0.2)
    if compile:
        model.eval()
        with torch.inference_mode():
            model = torch.compile(model)
    return model


def set_dropout(model, initial_value=0.0):
    warnings.warn(f"Setting dropout to {initial_value}")
    for k, v in list(model.named_modules()):
        if "drop" in k.split("."):
            parent_model = model
            for model_name in k.split(".")[:-1]:
                parent_model = getattr(parent_model, model_name)
            setattr(parent_model, "drop", nn.Dropout2d(p=initial_value))

    return model
