from .checkpoints import DOWNLOADABLE_MODELS
from .segmentation import segment, batch_segment


def list_models():
    return list(DOWNLOADABLE_MODELS.keys())