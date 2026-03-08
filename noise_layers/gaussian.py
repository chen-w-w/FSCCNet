import math

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussian_blur(nn.Module):
    def __init__(self,kernel,sigma):
        super(Gaussian_blur, self).__init__()
        self.kernel_size=int(kernel)
        self.sigma=float(sigma)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        gaussian_k = utils.gaussian_kernel(self.sigma, self.kernel_size)
        kernel_t = torch.FloatTensor(gaussian_k).unsqueeze(0).unsqueeze(0)

        self.register_buffer('gaussian_filter', kernel_t)

    def forward(self, noised_and_cover):
        # get the gaussian filter
        encode_image=noised_and_cover[0]
        batch_size,channel=encode_image.shape[0],encode_image.shape[1]
        assert encode_image.shape[1]==3

        kernel=self.gaussian_filter.expand(channel,1,self.kernel_size,self.kernel_size).to(self.device)

        padding = (self.kernel_size - 1) // 2

        noised_and_cover[0]=F.conv2d(encode_image,kernel,stride=1,padding=padding,groups=3)
        return noised_and_cover

    def __repr__(self):
        return f"Gaussian_blur({self.kernel_size},{self.sigma})"









        