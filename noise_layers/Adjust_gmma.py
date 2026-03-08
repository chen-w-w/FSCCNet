import torch.nn as nn
import torch
from kornia.enhance import AdjustGamma

class Adjust_gamm(nn.Module):
    def __init__(self, gamma, gain=1.0):

        super(Adjust_gamm, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def forward(self, noised_and_cover):

        encoded = (noised_and_cover[0]).clone()

        encoded = AdjustGamma(gamma=self.gamma, gain=self.gain)(encoded)

        noised_and_cover[0] = encoded
        return noised_and_cover
