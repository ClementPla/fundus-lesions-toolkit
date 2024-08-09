import torchseg
from huggingface_hub import get_collection
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import list_repo_refs
import torch.nn as nn
from typing import List, Union
from fundus_lesions_toolkit.constants import Dataset

ROOT_HF = "ClementP/fundus-lesions-segmentation-"


class HuggingFaceModel(PyTorchModelHubMixin, nn.Module):
    def __init__(self, arch: str, encoder: str, in_channels: int = 3, classes: int = 5):
        super().__init__()
        self.model = torchseg.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )


def download_model(
    arch, encoder_name, train_datasets: Union[Dataset, List[Dataset]] = Dataset.ALL
):
    if isinstance(train_datasets, tuple):
        train_datasets = list(train_datasets)

    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]
    train_datasets = list(set(train_datasets))
    train_datasets.sort()

    if Dataset.ALL in train_datasets:
        model = HuggingFaceModel.from_pretrained(
            f"{ROOT_HF}{arch}_{encoder_name}"
        ).model
    else:
        model = HuggingFaceModel.from_pretrained(
            f"{ROOT_HF}{arch}_{encoder_name}", revision="_".join(train_datasets)
        ).model

    return model


def list_models():
    collection = get_collection(ROOT_HF + "665f02dcdddc3d53c6e00274")
    print("Architecture | \033[94m Encoder | \033[92m Variants")
    for item in collection.items:
        if item.item_type == "model":
            name = item.item_id.split(ROOT_HF)[1]
            arch = name.split("_")[0]
            encoder = "_".join(name.split("_")[1:])
            branches = list_repo_refs(item.item_id).branches
            print(
                "\033[1m" + arch,
                "\033[94m" + encoder,
                f"\033[92m ({len(branches)} variants)",
            )
