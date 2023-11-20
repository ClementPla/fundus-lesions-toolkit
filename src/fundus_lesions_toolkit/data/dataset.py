from fundus_lesions_toolkit.utils.images import autocrop_fundus
import nntools.dataset as D
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nntools.dataset import nntools_wrapper

@nntools_wrapper
def autocrop(image, mask=None):
    h, w, c = image.shape
    image, roi, margins = autocrop_fundus(image)
    output = {'image':image, 'roi':roi}
    
    if mask is not None:
        mask = mask[margins[0]:h-margins[1], margins[2]:w-margins[3]]
        output['mask'] = mask
    return output

def get_dataset_from_folder(img_folder,
                        resolution=1536,
                        autopreprocess=True,
                        recursive_search=True,):
    
    dataset = D.ImageDataset(img_folder,shape=(resolution,resolution),
                             keep_size_ratio=True, 
                             recursive_loading=recursive_search,
                             auto_pad=True,)
    
    if autopreprocess:
        composer = D.Composition()
        composer.add(autocrop)
        resizing_ops = A.Compose([A.LongestMaxSize(max_size=resolution, always_apply=True),
                                  A.PadIfNeeded(min_height=resolution, min_width=resolution, always_apply=True,
                                                value=0, mask_value=0)],
                                 additional_targets={'roi':'mask'})
        composer.add(resizing_ops)
        composer.add(A.Compose([[A.Normalize(always_apply=True), ToTensorV2(transpose_mask=True)]]))
        
        dataset.composer = composer
    
    return dataset
    
        

    
    
    