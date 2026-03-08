import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.Dense_block import Bottleneck
from model.ConvNet import ConvBNRelu
from model.SENet import SENet,SENet_decoder
import numpy as np
from model.FSCC_Conv import FSCC_Conv

class Decoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7,
                         padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, config: HiDDenConfiguration, blocks=4):
        super(Decoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.channels = config.decoder_channels     # 64
        message_length = config.message_length

        stride_blocks = int(np.log2(self.H // int(np.sqrt(message_length))))    # 4
        keep_blocks = max(blocks - stride_blocks, 0)                            # 0

        self.first_layer = nn.Sequential(self.conv2(3, self.channels),
                                         nn.BatchNorm2d(self.channels),
                                         nn.LeakyReLU(inplace=True))

        self.second_layer = nn.Sequential(FSCC_Conv(self.channels,self.channels,L=2),
                                          nn.BatchNorm2d(self.channels),
                                          nn.LeakyReLU(inplace=True))

        self.third_layer = nn.Sequential(FSCC_Conv(self.channels * 2, self.channels,L=1),
                                         nn.BatchNorm2d(self.channels),
                                         nn.LeakyReLU(inplace=True))

        self.fourth_layer = nn.Sequential(FSCC_Conv(self.channels * 3, self.channels,L=0),
                                          nn.BatchNorm2d(self.channels),
                                          nn.LeakyReLU(inplace=True))
        self.fifth_layer = nn.Sequential(self.conv2(self.channels, self.channels),
                                         nn.BatchNorm2d(self.channels),
                                         nn.LeakyReLU(inplace=True))

        # decode

        self.six_layers = nn.Sequential(
            ConvBNRelu(self.channels, self.channels),
            # [B, channels, H, W] - > [B, channels * (2 ^ stride_blocks), H / (2 ^ stride_blocks), W / (2 ^ stride_blocks)]
            SENet_decoder(self.channels, self.channels, blocks=stride_blocks + 1),
            ConvBNRelu(self.channels * (2 ** stride_blocks), self.channels),
        )
        self.keep_layers = SENet(self.channels, self.channels, blocks=keep_blocks)

        self.final_layer = nn.Sequential(
            ConvBNRelu(self.channels, 1),
            # nn.Sigmoid()
        )

    def forward(self, image_with_wm):
        # stage1
        feature0 = self.first_layer(image_with_wm)
        feature1 = self.second_layer(feature0)
        feature2 = self.third_layer(torch.cat([feature0, feature1], dim=1))
        feature3 = self.fourth_layer(torch.cat([feature0, feature1, feature2], dim=1))  # [B, 24, 128, 128]
        feature4 = self.fifth_layer(feature3)   # [B, 24, 128, 128]

        # stage2
        x = self.six_layers(feature4)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        x = x.view(x.shape[0], -1)
        return x
