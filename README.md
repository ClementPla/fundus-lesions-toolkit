<p align="center">
    <img src="imgs/mosaic.png" width="256px"/>
</p>


# Toolkit for Lesions Segmentation in Fundus Images

This library offers a set of models (with pretrained weights) for the segmentation of lesions in fundus images.
As of now, four lesions are segmented

    1. Cotton Wool Spot
    2. Exudates
    3. Hemmorrhages
    4. Microaneurysms

    

## Installation

```bash
pip install fundus_lesions_toolkit
```

## Basic use case

Check the [notebooks](notebooks/) for detailed examples.

```python
import cv2
from fundus_lesions_toolkit.models import segment
img = cv.imread(img_path)
lesion = segment(img)
```