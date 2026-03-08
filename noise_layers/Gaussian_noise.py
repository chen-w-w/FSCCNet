import numpy as np
import torch.nn as nn
import torch

# class Gaussian_Noise(nn.Module):
#     def __init__(self,mean,sigma):
#         super(Gaussian_Noise, self).__init__()
#         self.mean=float(mean)
#         self.sigma=float(sigma)
#         self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#     def forward(self, noise_and_cover):
#         encode_image=noise_and_cover[0]
#         B,C,H,W=encode_image.size()
#         noise = np.clip(np.random.normal(self.mean,self.sigma , (B,1,H, W)), 0, 1)
#         noise=torch.tensor(noise,device=self.device)
#         for i in range(C):
#             encode_image[:,i:,:,:,]=encode_image[:,i:,:,:,]+noise
#         noise_and_cover[0]=encode_image
#         return noise_and_cover
#
#     def __repr__(self):
#         return f"Gaussian_Noise({self.mean},{self.sigma})"
class Gaussian_Noise(nn.Module):
    def __init__(self, mean, sigma):
        super(Gaussian_Noise, self).__init__()
        self.mean = float(mean)
        self.sigma = float(sigma)


    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        B, C, H, W = encode_image.size()

        noise = torch.normal(self.mean, self.sigma, size=(B, 1, H, W)).type_as(encode_image)

        noised_image = encode_image + noise

        noised_image = torch.clamp(noised_image, -1.0, 1.0)

        noise_and_cover[0] = noised_image
        return noise_and_cover

    def __repr__(self):
        return f"Gaussian_Noise({self.mean},{self.sigma})"