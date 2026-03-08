import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.FrequencyConv import FrequencyConv
from model.MultiConv_BiFPN import MultiConv_BiFPN

class FSCC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, L=1):
        super(FSCC_Conv, self).__init__()

        self.spatial_path = MultiConv_BiFPN(in_channels, out_channels, L)
        self.frequency_path = FrequencyConv(in_channels, out_channels, L)

        self.cross_domain_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        s = self.spatial_path(x)
        f = self.frequency_path(x)

        # Cross domain attention fusion
        concat_feat = torch.cat([s, f], dim=1)
        domain_weights = self.cross_domain_fusion(concat_feat)

        spatial_weight = domain_weights[:, 0:1]
        freq_weight = domain_weights[:, 1:2]

        fused = s * spatial_weight + f * freq_weight

        return self.output_conv(fused)
