# FSCCNet

Official PyTorch implementation of
**"FSCCNet: Robust Image Watermarking via Multi-Scale Collaborative Convolution and Perceptual Embedding Control."**

FSCCNet is a deep image-watermarking framework that **decouples robustness construction from perceptual quality control**:

- **FSCC Block** (Feature-Space Collaborative Convolution) builds robust watermark
  representations in the latent feature space through two complementary paths —
  a **Multi-Scale Aggregation (MSA)** path and a **Global Invariance and
  Recalibration (GIR)** path — fused by an **Adaptive Channel Gating (ACG)** mechanism.
- **Multi-Scale Dilated Attention (MSDA)** acts in the pixel space as an
  embedding-strength modulator, producing a content-adaptive spatial mask that
  concentrates watermark energy in texture-rich regions where the human visual
  system is least sensitive.

On COCO, ImageNet, DIV2K and CelebA-HQ, FSCCNet reaches an average bit-extraction
accuracy above 99% under standard single-distortion settings, with **PSNR 41.54 dB**
and **SSIM 0.9902**.

## Repository structure

```
model/
├── FSCCNet.py         # Top-level model wrapper (encoder / decoder / discriminator)
├── encoder.py         # Watermark encoder (FSCC U-Net + MSDA embedding mask)
├── decoder.py         # Watermark decoder
├── discriminator.py   # Adversarial discriminator
├── FSCC_Conv.py       # FSCC Block  =  MSA + GIR + ACG
├── MSA.py             # Multi-Scale Aggregation path
├── GIR.py             # Global Invariance and Recalibration path
├── Attention.py       # Multi-Scale Dilated Attention (MSDA) modules
└── ...                # SE / dense / conv building blocks
noise_layers/          # Differentiable distortion layers (JPEG, noise, blur, crop, ...)
options.py             # Configuration schema
SSIM.py / vgg_loss.py  # Loss / quality metrics
```

> **Note.** The training scripts (`main.py`, `train.py`, run scripts and training
> utilities) are **temporarily withheld** and will be released after publication.
> The module definitions above are provided for reference and review.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10 and torchvision
- timm
- numpy, Pillow

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2026fsccnet,
  title   = {FSCCNet: Robust Image Watermarking via Multi-Scale Collaborative
             Convolution and Perceptual Embedding Control},
  author  = {Chen, Jianhao and Xu, Xiaolong},
  year    = {2026}
}
```

## TODO
The traning code will be coming soon!

## License

Released under the MIT License. See [LICENSE](LICENSE).