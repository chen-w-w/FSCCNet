import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class MSA(nn.Module):
    def conv_block(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def __init__(self, in_channels, out_channels, L=1):
        super(MSA, self).__init__()

        assert in_channels % 12 == 0, "channels must be divisible by 12"

        c1 = in_channels // 3
        c2 = in_channels // 3 * 4
        c3 = in_channels // 12

        self.high_conv = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            nn.PixelShuffle(2),
            self.conv_block(c3, c3, kernel_size=5 + 2 * L, padding=2 + L),
            nn.PixelUnshuffle(2),
        )

        self.mid_conv = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            self.conv_block(c1, c1, kernel_size=3 + 2 * L, padding=1 + L),
        )

        self.low_conv = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            nn.PixelUnshuffle(2),
            self.conv_block(c2, c2, kernel_size=1 + 2 * L, padding=L),
            nn.PixelShuffle(2),
        )

        self.w = nn.Parameter(torch.ones(3),requires_grad=True)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        H = self.high_conv(x)
        M = self.mid_conv(x)
        L = self.low_conv(x)

        w = F.relu(self.w)
        weight = w / (w.sum() + 1e-4) * 3

        fused = torch.cat([weight[0] * H, weight[1] * M, weight[2] * L],dim=1)
        fused = self.fuse(fused)

        return self.final_conv(x + fused)
