
# Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling (HCFlow, ICCV2021)

This repository is the official PyTorch implementation of Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling
([arxiv](https://arxiv.org/pdf/2108.05301.pdf), [supp](https://github.com/JingyunLiang/HCFlow/releases/tag/v0.0)).


:rocket:  :rocket:  :rocket: **News**: 
 - Aug. 17, 2021: See our recent work for spatially variant kernel estimation: [Mutual Affine Network for Spatially Variant Kernel Estimation in Blind Image Super-Resolution (MANet), ICCV2021](https://github.com/JingyunLiang/MANet)
 - Aug. 17, 2021: See our recent work for real-world image SR: [Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (BSRGAN), ICCV2021](https://github.com/cszn/BSRGAN)
 - Aug. 17, 2021: See our previous flow-based work: *[Flow-based Kernel Prior with Application to Blind Super-Resolution (FKP), CVPR2021](https://github.com/JingyunLiang/FKP).*
 ---

> Normalizing flows have recently demonstrated promising results for low-level vision tasks. For image super-resolution (SR), it learns to predict diverse photo-realistic high-resolution (HR) images from the low-resolution (LR) image rather than learning a deterministic mapping. For image rescaling, it achieves high accuracy by jointly modelling the downscaling and upscaling processes. While existing approaches employ specialized techniques for these two tasks, we set out to unify them in a single formulation. In this paper, we propose the hierarchical conditional flow (HCFlow) as a unified framework for image SR and image rescaling. More specifically, HCFlow learns a bijective mapping between HR and LR image pairs by modelling the distribution of the LR image and the rest high-frequency component simultaneously. In particular, the high-frequency component is conditional on the LR image in a hierarchical manner. To further enhance the performance, other losses such as perceptual loss and GAN loss are combined with the commonly used negative log-likelihood loss in training. Extensive experiments on general image SR, face image SR and image rescaling have demonstrated that the proposed HCFlow achieves state-of-the-art performance in terms of both quantitative metrics and visual quality.
><p align="center">
  > <img height="120" src="./illustrations/computation_graph.png">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img height="120" src="./illustrations/architecture.png">
</p>

## Requirements
- Python 3.7, PyTorch == 1.7.1
- Requirements: opencv-python, lpips, natsort, etc.
- Platforms: Ubuntu 16.04, cuda-11.0


```bash
cd HCFlow-master
pip install -r requirements.txt 
```

## Quick Run (1 minute)
To run the code with one command (without preparing data), run this command:
```bash
cd codes
# face image SR
python test_HCFLow.py --opt options/test/test_SR_CelebA_8X_HCFlow.yml

# general image SR
python test_HCFLow.py --opt options/test/test_SR_DF2K_4X_HCFlow.yml

# image rescaling
python test_HCFLow.py --opt options/test/test_Rescaling_DF2K_4X_HCFlow.yml
```
---

## Data Preparation
The framework of this project is based on [MMSR](https://github.com/open-mmlab/mmediting) and [SRFlow](https://github.com/andreas128/SRFlow). To prepare data, put training and testing sets in `./datasets` as `./datasets/DIV2K/HR/0801.png`. Commonly used SR datasets can be downloaded [here](https://github.com/xinntao/BasicSR/blob/master/docs/DatasetPreparation.md#common-image-sr-datasets). 
There are two ways for accerleration in data loading: First, one can use `./scripts/png2npy.py` to generate `.npy` files and use `data/GTLQnpy_dataset.py`. Second, one can use `.pklv4` dataset (*recommended*) and use `data/LRHR_PKL_dataset.py`. Please refer to [SRFlow](https://github.com/andreas128/SRFlow#dataset-how-to-train-on-your-own-data) for more details. Prepared datasets can be downloaded [here](http://data.vision.ee.ethz.ch/alugmayr/SRFlow/datasets.zip).

## Training

To train HCFlow for general image SR/ face image SR/ image rescaling, run this command:

```bash
cd codes

# face image SR
python train_HCFLow.py --opt options/train/train_SR_CelebA_8X_HCFlow.yml

# general image SR
python train_HCFLow.py --opt options/train/train_SR_DF2K_4X_HCFlow.yml

# image rescaling
python train_HCFLow.py --opt options/train/train_Rescaling_DF2K_4X_HCFlow.yml
```
All trained models can be downloaded from [here](https://github.com/JingyunLiang/HCFlow/releases/tag/v0.0).


## Testing

Please follow the **Quick Run** section. Just modify the dataset path in `test_HCFlow_*.yml`.

## Results
We achieved state-of-the-art performance on general image SR, face image SR and image rescaling. 
> <img height="400" src="./illustrations/face_result.png">
> 
For more results, please refer to the [paper](https://arxiv.org/abs/2108.05301) and [supp](https://github.com/JingyunLiang/HCFlow/releases/tag/v0.0) for details. 

## Citation
    @inproceedings{liang21hcflow,
      title={Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling},
      author={Liang, Jingyun and Lugmayr, Andreas and Zhang, Kai and Danelljan, Martin and Van Gool, Luc and Timofte, Radu},
      booktitle={IEEE Conference on International Conference on Computer Vision},
      year={2021}
    }


## License & Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [MMSR](https://github.com/open-mmlab/mmediting), [SRFlow](https://github.com/andreas128/SRFlow), [IRN](https://github.com/pkuxmq/Invertible-Image-Rescaling) and [Glow-pytorch](https://github.com/chaiyujin/glow-pytorch). Please also follow their licenses. Thanks for their great works.

