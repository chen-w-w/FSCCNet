import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class FrequencyConv(nn.Module):
    def conv_block(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        )

    def __init__(self, in_channels, out_channels, L=1):
        super(FrequencyConv, self).__init__()
        self.L = L

        assert in_channels % 12 == 0, "channels must be divisible by 12"

        c1 = in_channels // 3
        c2 = in_channels // 3 * 4
        c3 = in_channels // 12

        self.high_freq = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            self.conv_block(c1, c1, kernel_size=7 + 2 * L, padding=3 + L),

            nn.Conv2d(c1, c1, 1),
            nn.Sigmoid()
        )

        self.mid_conv1 = self.conv_block(in_channels, c1, kernel_size=1)
        self.mid_conv2 = self.conv_block(c1, c1, kernel_size=3 + 2 * L, padding=1 + L)

        self.low_freq = nn.Sequential(
            self.conv_block(in_channels, c1, kernel_size=1),
            nn.AvgPool2d(4),
            self.conv_block(c1, c2, kernel_size=1 + 2 * L, padding=L),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            self.conv_block(c2, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 频域特征融合
        self.freq_fusion = nn.Sequential(
            self.conv_block(c1 * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 最终输出
        self.final_conv = self.conv_block(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        high_freq = self.high_freq(x)  # [B, C/3, H, W]

        mid_feat = self.mid_conv1(x)  # [B, C/3, H, W]
        mid_feat = F.adaptive_avg_pool2d(mid_feat, 8)  # [B, C/3, 8, 8]
        mid_feat = self.mid_conv2(mid_feat)  # [B, C/3, 8, 8]
        mid_freq = F.interpolate(mid_feat, size=(H, W), mode='bilinear', align_corners=False)
        mid_freq = torch.tanh(mid_freq)  # [B, C/3, H, W]

        low_freq = self.low_freq(x)  # [B, C/3, H, W]

        freq_features = torch.cat([high_freq, mid_freq, low_freq], dim=1)
        freq_output = self.freq_fusion(freq_features)  # [B, C, H, W]


        final_output = self.final_conv(freq_output)

        return final_output