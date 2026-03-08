import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float


class Resize(nn.Module):
    def __init__(self, resize_ratio_range, interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        original_height, original_width = noised_image.shape[2], noised_image.shape[3]

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)


        resized_image = F.interpolate(
            noised_image,
            scale_factor=resize_ratio,
            mode=self.interpolation_method,
            align_corners=False
        )

        restored_image = F.interpolate(
            resized_image,
            size=(original_height, original_width),
            mode=self.interpolation_method,
            align_corners=False
        )

        noised_and_cover[0] = restored_image
        return noised_and_cover

    def __repr__(self):
        return f"Resize(({self.resize_ratio_min},{self.resize_ratio_max}))"