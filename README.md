<p align="center">
    <img src="imgs/header.png" width="800px"/>
</p>


# Lesions Segmentation Toolkit for Fundus Images

This library offers a set of models (with pretrained weights) for the segmentation of lesions in fundus images.
As of now, four lesions are segmented:

    1. Cotton Wool Spot
    2. Exudates
    3. Hemmorrhages
    4. Microaneurysms

## Installation

```bash
pip install .
```



## Basic use

Check the [notebooks](notebooks/) for detailed examples.

```python
from fundus_lesions_toolkit.models import segment
from fundus_lesions_toolkit.constants import DEFAULT_COLORS, LESIONS
from fundus_lesions_toolkit.utils.images import open_image
from fundus_lesions_toolkit.utils.visualization import plot_image_and_mask

img = open_image(img_path)
pred = segment(img, device='cpu', weights='ALL')
plot_image_and_mask(img, pred, alpha=0.8, title='My segmentation', colors=DEFAULT_COLORS, labels=LESIONS)
```

## Models available

Currently, only a single model is made available (`unet` with a `resnest50` encoder). More will come regularly.

## Variants

Models are trained with different publicly available datasets:
1. [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
2. [MESSIDOR](https://www.adcis.net/fr/logiciels-tiers/messidor-fr/)
3. [DDR](https://github.com/nkicsl/DDR-dataset)
4. [FGADR](https://csyizhou.github.io/FGADR/)
5. [RETINAL-LESIONS](https://github.com/WeiQijie/retinal-lesions)

It also includes models trained with all the data combined.


## Training

This library is only a hub to access pure PyTorch trained models. 