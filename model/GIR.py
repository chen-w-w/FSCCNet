import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class GIR(nn.Module):
    def conv_block(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        )

    def __init__(self, in_channels, out_channels, L=1):
        super(GIR, self).__init__()
        self.L = L

        assert in_channels % 12 == 0, "channels must be divisible by 12"

        c1 = in_channels // 3
        c2 = in_channels // 3 * 4
        c3 = in_channels // 12

        self.large_kernel_attn = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            self.conv_block(c1, c1, kernel_size=7 + 2 * L, padding=3 + L),

            nn.Conv2d(c1, c1, 1),
            nn.Sigmoid()
        )

        self.global_reduce = self.conv_block(in_channels, c1, kernel_size=1)
        self.global_conv = self.conv_block(c1, c1, kernel_size=3 + 2 * L, padding=1 + L)

        self.local_detail = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            nn.AvgPool2d(4),
            self.conv_block(c1, c2, kernel_size=1 + 2 * L, padding=L),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            self.conv_block(c2, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # GIR branch fusion (Global Invariance and Recalibration)
        self.gir_fusion = nn.Sequential(
            self.conv_block(c1 * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # final output
        self.final_conv = self.conv_block(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        large_attn = self.large_kernel_attn(x)  # [B, C/3, H, W]

        global_feat = self.global_reduce(x)  # [B, C/3, H, W]
        global_feat = F.adaptive_avg_pool2d(global_feat, 8)  # [B, C/3, 8, 8]
        global_feat = self.global_conv(global_feat)  # [B, C/3, 8, 8]
        global_branch = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        global_branch = torch.tanh(global_branch)  # [B, C/3, H, W]

        local = self.local_detail(x)  # [B, C/3, H, W]

        gir_features = torch.cat([large_attn, global_branch, local], dim=1)
        gir_output = self.gir_fusion(gir_features)  # [B, C, H, W]


        final_output = self.final_conv(gir_output)

        return final_output