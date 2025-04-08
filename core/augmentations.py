import math
import random
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.transforms.autoaugment import RandAugment, TrivialAugmentWide, AugMix
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image
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
        img = F.affine(
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
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
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
        "Brightness": (torch.linspace(0.0, 0.9, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "Color": (torch.linspace(0.0, 0.9, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "Contrast": (torch.linspace(0.0, 0.9, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "Sharpness": (torch.linspace(0.0, 0.9, mag_bins), torch.linspace(0.0, 1, prob_bins), True),
        "Posterize": (8 - (torch.arange(mag_bins) / ((mag_bins - 1) / 4)).round().int(), torch.linspace(0.0, 1, prob_bins), False),
        "Solarize": (torch.linspace(255.0, 0.0, mag_bins), torch.linspace(0.0, 1, prob_bins), False),
        "AutoContrast": (torch.tensor(0.0), torch.linspace(0.0, 1, prob_bins), False),
        "Equalize": (torch.tensor(0.0), torch.linspace(0.0, 1, prob_bins), False),
        }
  
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
        ops = all_ops[idx][0].tolist()
        all_magnitude = self.policy['magnitude_index'][idx]
        magnitude = all_magnitude[np.random.randint(all_magnitude.shape[0]),:]
        all_prob = self.policy['prob_index'][idx]
        prob = all_prob[np.random.randint(all_prob.shape[0]),:]
        for i, p in enumerate(ops):
            op_name = list(aug_space.keys())[int(p)]
            magnitudes, prob, signed = aug_space[op_name]
            magnitude_index = int(magnitude[i])
            prob_index = int(prob[i])
            if magnitudes.ndim > 0:
                magnitudes = magnitudes[:magnitude_index+1]
                m = float(magnitudes[random.randint(0, magnitudes.shape[0]-1)].item())
            else:
                m = 0.0
                # magnitude = float(magnitudes[magnitude_index].item())
            p = float(prob[prob_index].item())
            if signed and torch.randint(2, (1,)):
                m *= -1.0
            if torch.randn(1) < p:
                img = _apply_op(img, op_name, m, interpolation=self.interpolation, fill=fill)
        if self.resize:
            img = F.resize(img, self.resize_size)

        return img