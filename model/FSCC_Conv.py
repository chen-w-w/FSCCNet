import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.GIR import GIR
from model.MSA import MSA

class FSCC_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, L=1):
        super(FSCC_Conv, self).__init__()

        self.msa_path = MSA(in_channels, out_channels, L)
        self.gir_path = GIR(in_channels, out_channels, L)

        self.adaptive_channel_gating = nn.Sequential(
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
        s = self.msa_path(x)
        g = self.gir_path(x)

        # Adaptive Channel Gating (ACG)
        concat_feat = torch.cat([s, g], dim=1)
        gate_weights = self.adaptive_channel_gating(concat_feat)

        alpha_s = gate_weights[:, 0:1]
        alpha_g = gate_weights[:, 1:2]

        fused = s * alpha_s + g * alpha_g

        return self.output_conv(fused)