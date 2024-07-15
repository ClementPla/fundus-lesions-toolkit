from typing import Literal, Union, Tuple

import numpy as np
import torch
from fundus_lesions_toolkit.constants import (
    Dataset,
)
from fundus_lesions_toolkit.constants import LESIONS, lesions2names

from fundus_lesions_toolkit.models.segmentation import segment
from kornia.contrib import connected_components


Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]


def count_lesions(
    image: np.ndarray,
    size_threshold: int = 0,
    arch: Architecture = "unet",
    encoder: EncoderModel = "seresnext50_32x4d",
    train_datasets: Union[Dataset, Tuple[Dataset]] = Dataset.ALL,
    image_resolution=1024,
    autofit_resolution=True,
    reverse_autofit=True,
    mean=None,
    std=None,
    device: torch.device = "cuda",
    compile: bool = False,
) -> dict:
    pred = segment(
        image,
        arch=arch,
        encoder=encoder,
        train_datasets=train_datasets,
        image_resolution=image_resolution,
        autofit_resolution=autofit_resolution,
        reverse_autofit=reverse_autofit,
        mean=mean,
        std=std,
        device=device,
        compile=compile,
    )

    pred = pred.argmax(0)
    out = {}
    for i, lesion in enumerate(LESIONS):
        if i == 0:
            continue

        mask = pred == i
        ccl = connected_components(mask.unsqueeze(0).unsqueeze(0).float())

        unique, count = torch.unique(ccl, return_counts=True, sorted=True)
        nb_components = (
            count > size_threshold
        ).sum().item() - 1  # -1 to remove background

        out[lesions2names[lesion]] = nb_components

    return out
