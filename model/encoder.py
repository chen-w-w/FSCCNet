import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.Dense_block import Bottleneck
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from model.ConvNet import ConvBNRelu
from model.ExpandNet import ExpandNet
from model.SENet import SENet
from model.FrequencyConv import FrequencyConv
from model.FSCC_Conv import FSCC_Conv
from model.MultiConv_BiFPN import MultiConv_BiFPN
from model.Attention import *

class MP(nn.Module):
    def __init__(self, H, W, message_length, blocks=4, channels=24):
        super(MP, self).__init__()
        self.H = H
        self.W = W
        message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        message_se_blocks = max(blocks - message_convT_blocks, 1)

        self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet(channels, channels, blocks=message_se_blocks),
		)

        self.message_first_layer = SENet(channels, channels, blocks=blocks)
    def forward(self, message):
        # Message Processor
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        message_pre = self.message_pre_layer(message_image)
        message_feature = self.message_first_layer(message_pre)
        return message_feature

class Encoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=3, padding=1)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels

        self.watermarkProcessor = MP(self.H, self.W, config.message_length, blocks=4, channels=self.conv_channels)
        self.imageProcessor = self.conv2(3, self.conv_channels)

        self.initial_conv = nn.Conv2d(2 * self.conv_channels, self.conv_channels, 3, padding=1)

        self.encoder1 = nn.Sequential(
            FSCC_Conv(self.conv_channels, self.conv_channels, L=3),
        )

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            FSCC_Conv(self.conv_channels, 2 * self.conv_channels, L=2),
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            FSCC_Conv(2 * self.conv_channels, 4 * self.conv_channels,L=1)
        )

        self.decoder3 = nn.Sequential(
            FSCC_Conv(4 * self.conv_channels, 2 * self.conv_channels, L=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

        )

        self.decoder2 = nn.Sequential(
            FSCC_Conv(4 * self.conv_channels, self.conv_channels, L=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

        )

        self.decoder1 = nn.Sequential(
            FSCC_Conv(2 * self.conv_channels, self.conv_channels, L=3),
        )

        self.first_layer = nn.Sequential(
            self.conv2(3, 64)
        )

        self.dilate_stage = Dilatestage(
            dim=64,
            depth=2,
            num_heads=8,
            kernel_size=3,
            dilation=[1, 2],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            cpe_per_satge=False,
            cpe_per_block=False,
            downsample=True,
            merging_way='conv3_2'
        )

        self.pre_global_down = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.global_stage = Globalstage(
            dim=64 * 2,
            depth=2,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            cpe_per_satge=False,
            cpe_per_block=False,
            downsample=False,
            merging_way=None
        )

        self.global_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.conv2(64,64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.conv2(64, 24),
            nn.Softmax(dim=1)
        )

        self.final_layer = nn.Sequential(nn.Conv2d(24, 3, kernel_size=3, padding=1),
                                         )

    def forward(self, image, message):
        B, C, H, W = image.size()[0], image.size()[1], image.size()[2], image.size()[3]

        processed_message = self.watermarkProcessor(message)  # [B, 24, 128, 128]
        processed_image = self.imageProcessor(image)    # [B, 24, 128, 128]

        enc1 = self.encoder1(self.initial_conv(torch.cat((processed_image,processed_message),dim=1)))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        dec3 = self.decoder3(enc3)
        dec2 = self.decoder2(torch.cat((dec3,enc2),dim=1))
        dec1 = self.decoder1(torch.cat((dec2,enc1),dim=1))


        feature0 = self.first_layer(image)
        feature_local = self.dilate_stage(feature0)
        feature_local = self.pre_global_down(feature_local)
        feature_attention = self.global_stage(feature_local)
        feature_attention = self.global_upsample(feature_attention)
        feature_mask = (self.sixth_layer(feature_attention)) * 24
        feature = dec1 * feature_mask

        im_w = self.final_layer(feature)
        im_w = im_w + image
        return im_w










