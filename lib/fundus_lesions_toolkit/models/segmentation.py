import numpy as np
import torch
import os
from typing import Literal
from checkpoints import downloadable_models
Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]
TrainedOn = Literal["ALL"]


_last_model = (None, None)


def segment(x:np.ndarray, arch: Architecture, 
            encoder: EncoderModel,
            weights:TrainedOn='All',
            auto_crop=False, 
            device: torch.device="cuda"):
    
    global _last_model
    if _last_model[0] == (arch, encoder, weights):
        model = _last_model[1]
    else:
        model = segmentation_model().to(device=device)
        _last_model = ((arch, encoder, weights), model)

def list_models():
    return list(downloadable_models.keys())
    
def segmentation_model(arch:Architecture, encoder:EncoderModel, weights:TrainedOn):
    arch = arch.lower()
    weights = weights.lower()
    encoder = encoder.lower()
    
    assert (arch, encoder, weights) in downloadable_models.keys(), f"Wrong combinations of architecture, encoder and weights asked {(arch, encoder, weights)}. Call list_models() to see all configurations acceptable"
    import segmentation_models_pytorch as smp
    model = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None, in_channels=3, classes=5)
    
    url = downloadable_models[(arch, encoder, weights)]
    model_name = f'{arch}_{encoder}_{weights}.ckpt'
    model_dir = os.path.join(torch.hub.get_dir(), 'checkpoints/fundus_segmentation_toolkit/segmentation')
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', file_name=model_name, model_dir=model_dir)
    
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model