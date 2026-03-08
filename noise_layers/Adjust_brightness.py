import torch.nn as nn
import torch
from kornia.enhance import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,AdjustGamma
import math
from torchvision.transforms import ToPILImage
class Adjust_brightness(nn.Module):
    def __init__(self,factor):
        super(Adjust_brightness, self).__init__()
        self.factor=factor


    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustBrightness(brightness_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover


    def __repr__(self):
        return f"Adjust_brightness({self.factor})"