from .checkpoints import downloadable_models

def list_models():
    return list(downloadable_models.keys())