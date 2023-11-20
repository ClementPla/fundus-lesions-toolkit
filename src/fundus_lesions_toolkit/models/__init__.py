from fundus_lesions_toolkit.models.segmentation import segment, batch_segment
from fundus_lesions_toolkit.models.checkpoints import DOWNLOADABLE_MODELS


def list_models():
    return list(DOWNLOADABLE_MODELS.keys())