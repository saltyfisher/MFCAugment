import math
import random
import numpy as np
import cv2
import PIL
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image
from core.utils import trimap_generate
from torchvision.transforms import _functional_tensor as F_t

def generate_saliency_map(img):
    """
    生成图像的显著性图
    
    Args:
        image: 输入图像 (C, H, W)
    
    Returns:
        saliency_map: 显著性图 (H, W)
    """
   
    sailency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = sailency.computeSaliency(img)
    # 归一化到0-1范围
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    
    return saliency_map

def MM(img, superpixel_num, trimap_alpha, matting_method):
    if isinstance(img, Tensor):
        img = img.cpu().numpy()
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    superpixel_num = int(np.round(100*superpixel_num))
    # trimap_alpha = np.round(100*trimap_alpha)
    # img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    w, h, c = img.shape
    m_size = 64
    device = next(matting_method.parameters()).device
    # 超像素分割
    cluster = cv2.ximgproc.createSuperpixelSEEDS(
        h, w, c, 
        superpixel_num,
        num_levels=4 
    )        
    cluster.iterate(img, 4)
    segement = cluster.getLabels()
    # 显著图计算
    saliency_map = generate_saliency_map(img)
    # 选择超像素块
    unique_labels = np.unique(segement)
    weights = []
    for label in unique_labels:
        mask = segement == label
        weight = np.sum(saliency_map[mask])
        weights.append(weight)
    
    # 归一化权重
    weights = np.array(weights)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        # 如果所有权重都为0，使用均匀分布
        weights = np.ones_like(weights) / len(weights)
    
    # 根据权重随机选择超像素块
    selected_label = np.random.choice(unique_labels, p=weights)
    mask = segement == selected_label
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    cropped_region = img[min_row:max_row, min_col:max_col]
    # 获取抠图遮罩
    cropped_region = cv2.resize(cropped_region, (m_size, m_size), interpolation=cv2.INTER_CUBIC)
    cropped_saliency_map = generate_saliency_map(cropped_region)
    matting_input = torch.zeros(1, 4, m_size, m_size)
    matting_input[0, :c] = torch.from_numpy(cropped_region.transpose(2, 0, 1) / 255.)
    trimap = trimap_generate(cropped_region, cropped_saliency_map, trimap_alpha_threshold=trimap_alpha,trimap_gen='stats')
    trimap = trimap[np.newaxis, :, :]
    matting_input[0,:3] = torch.from_numpy(trimap/255.).to(device)
    with torch.no_grad():
        refined_mask = matting_method(matting_input.to(device))
    refined_mask = refined_mask.cpu().numpy()
    refined_mask[trimap == 0] = 0.0
    refined_mask[trimap == 255] = 1.0
    refined_mask = refined_mask * 255
    refined_mask = refined_mask.astype(np.uint8)
    ratio = (superpixel_num/100.)*np.sum(refined_mask/255.)/(refined_mask.shape[1]*refined_mask.shape[2])
    mask = np.zeros((w, h))
    refined_mask = refined_mask.squeeze()
    mask[min_row:max_row, min_col:max_col] = cv2.resize(refined_mask, (max_col-min_col, max_row-min_row), interpolation=cv2.INTER_CUBIC)
    mask = mask/255.
    mask = np.expand_dims(mask, axis=2)
    fg = img * mask
    cropped_fg = fg[min_row:max_row, min_col:max_col]
    pos = (min_row, max_row, min_col, max_col)
    return fg, cropped_fg, pos, mask, ratio


def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img= F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img = F.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = F.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img

def _apply_op_mm(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    affine_matrix = []
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img, affine_matrix = F.affine_mm(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img, affine_matrix = F.affine_mm(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        img, affine_matrix = F.affine_mm(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img, affine_matrix = F.affine_mm(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img, affine_matrix = F.rotate_mm(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img, affine_matrix

def augmentation_space(mag_bins: int=31, prob_bins: int=10, image_size: Tuple[int, int]=[320,320]) -> Dict[str, Tuple[Tensor, Tensor, bool]]:
    return {
        # 坐标空间操作
        "Identity": (torch.tensor(0.0), torch.linspace(0.0, 1, prob_bins), False),
        "ShearX": (torch.linspace(0.0, 1, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "ShearY": (torch.linspace(0.0, 1, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "TranslateX": (torch.linspace(0.0, image_size[1], mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "TranslateY": (torch.linspace(0.0, image_size[0], mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "Rotate": (torch.linspace(0.0, 180.0, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        # 颜色空间操作
        "Brightness": (torch.linspace(0.0, 2.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "Color": (torch.linspace(0.0, 2.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "Contrast": (torch.linspace(0.0, 2.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "Sharpness": (torch.linspace(0.0, 2.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "Posterize": (8 - (torch.arange(mag_bins) / ((mag_bins - 1) / 4)).round().int(), torch.linspace(0.0, 1, prob_bins), False),
        "Solarize": (torch.linspace(255.0, 0.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "AutoContrast": (torch.tensor(0.0), torch.linspace(0.0, 1, prob_bins), False),
        "Equalize": (torch.tensor(0.0), torch.linspace(0.0, 1, prob_bins), False),
        }

class MyRandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    # def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
    #     return {
    #         # op_name: (magnitudes, signed)
    #         "Identity": (torch.tensor(0.0), False),
    #         "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
    #         "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
    #         "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
    #         "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
    #         "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
    #         "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
    #         "Color": (torch.linspace(0.0, 0.9, num_bins), True),
    #         "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
    #         "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
    #         "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
    #         "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
    #         "AutoContrast": (torch.tensor(0.0), False),
    #         "Equalize": (torch.tensor(0.0), False),
    #     }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = augmentation_space(mag_bins=self.num_magnitude_bins, image_size=(height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, _, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


class MyTrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    # def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
    #     return {
    #         # op_name: (magnitudes, signed)
    #         "Identity": (torch.tensor(0.0), False),
    #         "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
    #         "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
    #         "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
    #         "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "Color": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
    #         "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
    #         "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
    #         "AutoContrast": (torch.tensor(0.0), False),
    #         "Equalize": (torch.tensor(0.0), False),
    #     }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = augmentation_space(self.num_magnitude_bins, image_size=(height, width))
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, _, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

class MyAugment(torch.nn.Module):
    def __init__(
        self,
        policy,
        mag_neigbor_range : int = 1,
        num_ops: int = 2,
        magnitude: int = 9,
        mag_bin: int = 31,
        prob_bin: int = 10,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        resize = False,
        resize_size = None
    ) -> None:
        super().__init__()
        self.mag_bin = mag_bin
        self.prob_bin = prob_bin
        self.interpolation = interpolation
        self.fill = fill
        self.num_ops = num_ops
        # self.policy = torch.t(torch.tensor(policy).reshape(-1,num_ops))
        self.policy = policy
        self.mag_neighbor_range = mag_neigbor_range
        self.magnitude = magnitude
        self.resize = resize
        self.resize_size = resize_size

    def get_policy(self, policy):
        self.policy = policy
        # for p in policy:
        #     op_index, magnitude, prob = list(*p)
        #     old_magnitude = list(aug_space.values())[op_index][0] 
        #     old_prob = list(aug_space.values())[op_index][1] 
        #     key = list(aug_space.keys())[op_index]
        #     if old_magnitude.ndim == 0:
        #         new_policy[key] = list(old_space.values())[op_index]
        #     else:
        #         new_magnitude = list(range(magnitude-self.mag_neighbor_range,magnitude+self.mag_neighbor_range+1))
        #         new_magnitude = [m for m in new_magnitude if m >=0 and m<self.num_magnitude_bins] 
        #         new_magnitude = torch.tensor([old_magnitude[m] for m in new_magnitude])
            
        #         if key in new_policy.keys():
        #             magnitude = new_policy[key][0]
        #             magnitude = torch.unique(torch.hstack((magnitude, new_magnitude)))
        #             new_policy[key] = (magnitude, new_policy[key][1])
        #         else:                
        #             new_policy[key] = (new_magnitude, old_space[key][1])
        # return new_policy
    
    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        aug_space = augmentation_space(self.mag_bin, self.prob_bin, (height, width))
        # for p in self.policy:
        #     op_index, magnitude_index, prob_index = map(int,p.tolist())
        #     op_name = list(aug_space.keys())[op_index]
        #     magnitudes, prob, signed = aug_space[op_name]
        #     if magnitudes.ndim > 0:
        #         magnitudes = magnitudes[:magnitude_index+1]
        #         magnitude = float(magnitudes[torch.randint(magnitudes.shape[0], (1,))].item())
        #         # magnitude = float(magnitudes[magnitude_index].item())
        #     else:
        #         magnitude = 0.0
        #     prob = float(prob[prob_index].item())
        #     if signed and torch.randint(2, (1,)):
        #         magnitude *= -1.0
        #     if torch.randn(1) < prob:
        #         img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        # if self.resize:
        #     img = F.resize(img, self.resize_size)
        img = self.augment_img(img, aug_space, fill)
        return img
    
    def augment_img(self, img, aug_space, fill):
        all_ops = self.policy['op_index']
        idx = np.random.randint(len(all_ops))
        ops = all_ops[idx].tolist()
        all_magnitude = self.policy['magnitude_index'][idx]
        if len(all_magnitude.shape) < 2:
            all_magnitude = all_magnitude[np.newaxis]
        magnitude = all_magnitude[np.random.randint(all_magnitude.shape[0])]
        if self.policy['prob_index'] != []:
            all_prob = self.policy['prob_index'][idx]
            if len(all_prob.shape) < 2:
                all_prob = all_prob[np.newaxis]
            prob = all_prob[np.random.randint(all_prob.shape[0])]
        for i, p in enumerate(ops):
            op_name = list(aug_space.keys())[int(p)]
            magnitudes, prob, signed = aug_space[op_name]
            magnitude_index = int(magnitude[i])
            if magnitudes.ndim > 0:
                # magnitudes = magnitudes[:magnitude_index+1]
                # m = float(magnitudes[random.randint(0, magnitudes.shape[0]-1)].item())
                m = float(magnitudes[magnitude_index])
            else:
                m = 0.0
                # magnitude = float(magnitudes[magnitude_index].item())
            if signed and torch.randint(2, (1,)):
                m *= -1.0
            try:
                prob_index = int(prob[i])
                p = float(prob[prob_index].item())
                if torch.randn(1) < p:
                    img = _apply_op(img, op_name, m, interpolation=self.interpolation, fill=fill)
            except:
                img = _apply_op(img, op_name, m, interpolation=self.interpolation, fill=fill)
        if self.resize:
            img = F.resize(img, self.resize_size)

        return img

class MyAugmentMM(torch.nn.Module):
    def __init__(
        self,
        policy,
        mag_neigbor_range : int = 1,
        num_ops: int = 2,
        magnitude: int = 9,
        mag_bin: int = 31,
        prob_bin: int = 10,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        resize = False,
        resize_size = None,
        post_augment = True
    ) -> None:
        super().__init__()
        self.mag_bin = mag_bin
        self.prob_bin = prob_bin
        self.interpolation = interpolation
        self.fill = fill
        self.num_ops = num_ops
        # self.policy = torch.t(torch.tensor(policy).reshape(-1,num_ops))
        self.policy = policy
        self.mag_neighbor_range = mag_neigbor_range
        self.magnitude = magnitude
        self.resize = resize
        self.resize_size = resize_size
        self.post_augment = post_augment

    def get_policy(self, policy):
        self.policy = policy
    
    def forward(self, img: Tensor, matting_method) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        aug_space = augmentation_space(self.mag_bin, self.prob_bin, (height, width))
        img, mask, ratio = self.augment_img(img, aug_space, fill, matting_method)
        return img, mask, ratio
    
    def augment_img(self, img, aug_space, fill, matting_method):
        # 取出MM的参数
        # self.policy['op_index'] = self.policy['op_index'][1:]
        # all_superpixel_num = self.policy['magnitude_index'][0]
        # all_trimap_alpha = self.policy['prob_index'][0]
        # superpixel_num = all_superpixel_num[np.random.randint(all_superpixel_num.shape[0])]
        # trimap_alpha = all_trimap_alpha[np.random.randint(all_trimap_alpha.shape[0])]
        
        # mask = torch.from_numpy(mask)
        img = img.permute(1, 2, 0)
        geo_trans = []
        all_ops = self.policy['op_index']
        if all_ops.shape == ():
            all_ops = [all_ops[np.newaxis]]
        idx = np.random.randint(len(all_ops))
        ops = all_ops[idx].tolist()
        all_magnitude = self.policy['magnitude_index'][idx]
        if len(all_magnitude.shape) < 2:
            all_magnitude = all_magnitude[np.newaxis]
        magnitude = all_magnitude[np.random.randint(all_magnitude.shape[0])]
        if len(self.policy['prob_index']) > 0:
            all_prob = self.policy['prob_index'][idx]
            if len(all_prob.shape) < 2:
                all_prob = all_prob[np.newaxis]
            prob = all_prob[np.random.randint(all_prob.shape[0])]
        affine_matrix = []
        for i, p in enumerate(ops):
            if p == 100:
                superpixel_num = magnitude[i]
                trimap_alpha = prob[i]
                fg, cropped_fg, pos, mask, ratio = MM(img, 
                                    superpixel_num=superpixel_num,
                                    trimap_alpha=trimap_alpha,
                                    matting_method=matting_method
                                    )
                mask = torch.FloatTensor(mask).permute(2, 0, 1)
                img = img.permute(2, 0, 1).to(torch.uint8)
            else:
                # p = 11
                if self.post_augment:
                    signed = False
                    op_name = list(aug_space.keys())[int(p)]
                    magnitudes, probs, signed = aug_space[op_name]
                    magnitude_index = int(magnitude[i])
                    if magnitudes.ndim > 0:
                        # magnitudes = magnitudes[:magnitude_index+1]
                        # m = float(magnitudes[random.randint(0, magnitudes.shape[0]-1)].item())
                        m = float(magnitudes[magnitude_index].item())
                    else:
                        m = 0.0
                        # magnitude = float(magnitudes[magnitude_index].item())
                    if signed and torch.randint(2, (1,)):
                        m *= -1.0
                    if i+1 <= len(prob):
                        prob_index = int(prob[i])
                        p = float(prob[prob_index].item())
                        if torch.randn(1) < p:
                            img, affine_matrix = _apply_op_mm(img, op_name, m, interpolation=self.interpolation, fill=fill)
                    else:
                        img, affine_matrix = _apply_op_mm(img, op_name, m, interpolation=self.interpolation, fill=fill)
                    if affine_matrix != []:
                        geo_trans.append([op_name, affine_matrix])
        if geo_trans != [] and self.post_augment:
            for op_name, affine_matrix in geo_trans:
                if op_name == 'Rotate':
                    mask = F_t.rotate(mask, matrix=affine_matrix)
                else:
                    mask = F_t.affine(mask, matrix=affine_matrix)

        if self.resize:
            img = F.resize(img, self.resize_size)
        mask = mask.to(img.device)
        img = img * mask
        return img, mask, ratio
