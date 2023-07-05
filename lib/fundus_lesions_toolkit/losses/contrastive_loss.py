"""
Pytorch custom Implementation of the different losses proposed in the paper:
Contrastive Learning for Label Efficient Semantic Segmentation
https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Contrastive_Learning_for_Label_Efficient_Semantic_Segmentation_ICCV_2021_paper.pdf
By Zhao et al.

"""
from enum import IntEnum, auto
from random import random
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF



class ContrastiveLoss(IntEnum):
    SINGLE_IMAGE = auto()
    CROSS_IMAGES = auto()
    
    
def within_image_loss(img: torch.Tensor, img_jittered: torch.Tensor, masks: torch.Tensor,
                      tau: float, epsilon: float = 1e-5, **kwargs) -> torch.Tensor:
    """
    :param img: Tensor DxHxW Features vector associated to the original image
    :param img_jittered: Same, but from the distorted image (color jittering)
    :param masks: Tensor CxHxW Groundtruth (same for both image because the distortion is only color wise). Each layer in
    the first dimension is a label.
    :param tau Temperature parameter
    :param epsilon: for numerical stability (not in the paper)
    :param kwargs:
    :return:
    """
    img_flatten = img.flatten(1, 2).transpose(0, 1)  # DxN -> NxD
    N, D = img_flatten.shape
    jit_flatten = img_jittered.flatten(1, 2)  # DxN
    quadratic_matrix = torch.exp(torch.matmul(img_flatten, jit_flatten) / tau)  # NxN: e_{pq}^{IÎ}
    normalization_matrix = quadratic_matrix.sum(1, keepdim=True)  # Nx1 \Sigma e_{pq}^{IÎ}
    loss = 0
    for l in masks:
        l = l.flatten() > 0
        if not l.any():
            continue
        quad_L = quadratic_matrix[l][:, l]  # LxL
        loss += -torch.log(quad_L / (normalization_matrix[l] + epsilon)).sum() / l.sum()
    return loss / N


def cross_image_loss(img_a: torch.Tensor, img_a_jit: torch.Tensor,
                     img_b: torch.Tensor, masks_a: torch.Tensor, masks_b: torch.Tensor,
                     tau: float, epsilon: float = 1e-5, **kwargs) -> torch.Tensor:
    """
    :param img_a: Tensor DxHxW Features vector associated to the original image
    :param img_a_jit: Same, but from the distorted version of the image
    :param img_b: Tensor DxHxW Features vector associated to another image (but from the same batch)
    :param masks_a: Tensor CxHxW Gt for image A
    :param masks_b: Tensor CxHxW Gt for image B
    :param tau Temperature parameter
    :param epsilon: for numerical stability (not in the paper)
    :param kwargs:
    :return:
    """
    img_a_flatten = img_a.flatten(1, 2).transpose(0, 1)
    N, D = img_a_flatten.shape
    img_a_jit_flatten = img_a_jit.flatten(1, 2)
    img_b_flatten = img_b.flatten(1, 2)
    within_img_pairing = torch.exp(torch.matmul(img_a_flatten, img_a_jit_flatten) / tau)  # NxN: e_{pq}^{IÎ}
    cross_img_pairing = torch.exp(torch.matmul(img_a_flatten, img_b_flatten) / tau)  # NxN: e_{pq}^{IĴ}
    normalization_within_img = within_img_pairing.sum(1)  # Nx1 \Sigma e_{pq}^{IÎ}
    loss = 0
    for l_a, l_b in zip(masks_a, masks_b):
        l_a = l_a.flatten() > 0
        l_b = l_b.flatten() > 0
        if not l_a.any() or not l_b.any():
            continue
        e_IJ = cross_img_pairing[l_a][:, l_b]
        l1 = torch.log(within_img_pairing[l_a][:, l_a] / (normalization_within_img[l_a] + e_IJ.sum(1) + epsilon)).sum()
        l2 = torch.log(e_IJ / torch.unsqueeze(normalization_within_img[l_a] + e_IJ.sum(1) + epsilon, 1)).sum()
        loss += -(l1 + l2) / (l_a.sum() + l_b.sum())
    return loss / N


def random_jitter(batch_img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  jitter_proba=0.8, max_hue=0.08):
    proba = random()  # Uniform sort between [0, 1]
    if proba < jitter_proba:
        batch_std = torch.Tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(batch_img)
        batch_mean = torch.Tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(batch_img)
        batch_img = (batch_img * batch_std) + batch_mean  # Bring back to the [0,1] range.
        proba = max_hue * proba
        batch_img = TF.adjust_hue(batch_img, proba)
        batch_img = (batch_img - batch_mean) / batch_std
    return batch_img


def contrastive_loss_from_batch(model, imgs, gts, loss_type=ContrastiveLoss.SINGLE_IMAGE, **c):
    gts = F.interpolate(gts.float(), size=c['size']).long()
    bg_class = (torch.max(gts, 1, keepdim=True)[0] == 0).long()
    gts = torch.cat([gts, bg_class], 1)
    imgs_jit = random_jitter(imgs.clone())
    b, d, h, w = imgs.shape
    full_features = model(torch.cat([imgs, imgs_jit], 0))
    full_features = F.interpolate(full_features, size=c['size'])
    features = full_features[:b]
    features_jit = full_features[b:]
    loss = 0
    if loss_type == ContrastiveLoss.SINGLE_IMAGE:
        for feat, feat_jit, gt in zip(features, features_jit, gts):
            loss += within_image_loss(feat, feat_jit, gt, **c)
    if loss_type == ContrastiveLoss.CROSS_IMAGES:
        for i, (feat, feat_jit, gt) in enumerate(zip(features, features_jit, gts)):
            index = b - (i + 1)
            if i == index:
                index = 0
            feat_b = features[index]
            gt_b = gts[index]
            loss += cross_image_loss(feat, feat_jit, feat_b, gt, gt_b, **c)
    return loss / b
