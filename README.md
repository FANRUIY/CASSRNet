# CASSRNet
**Enhancing Image Super-Resolution via Complexity-Aware and Dynamic Inference Networks**  
*(Based on and inspired by ClassSR and RCAN frameworks)*

[Paper (Preprint)](https://arxiv.org/) — Ruiyang Fan, Guobing Sun  
Heilongjiang University, Harbin, China  

---

## 🌟 Abstract

In this work, we propose **CASSRNet** — a *Complexity-Aware Split Super-Resolution Network* designed to balance reconstruction fidelity and computational efficiency in single-image super-resolution (SISR).  
Traditional SR models such as RCAN and EDSR use fixed inference pipelines, leading to redundant computation on simple regions.  
CASSRNet introduces **complexity-driven dynamic inference**, enabling adaptive branch selection based on input difficulty.

Key highlights:
- **Efficient Architecture:** Incorporates *Depthwise-Separable Convolutional Residual Attention Blocks (DCSAB)*, *Hybrid Channel–Spatial Attention*, and *Window Attention* for stronger feature representation.
- **Complexity-Aware Training:** Builds a *complexity-annotated DF2K dataset* with instance segmentation and complexity scoring, dividing samples into *simple*, *medium*, and *complex* levels.
- **Dynamic Inference:** Uses a *three-branch selection strategy* to balance accuracy and efficiency adaptively.

Experiments show that CASSRNet achieves state-of-the-art reconstruction quality with an **average FLOPs reduction of 27.3%**, while maintaining high fidelity in fine-textured regions.

---

## 🔧 Dependencies

- Python >= 3.6 (Recommend [Anaconda](https://www.anaconda.com/download))
- [PyTorch >= 1.5.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Python packages:
  ```bash
  pip install numpy opencv-python lmdb tqdm scikit-image tensorboardX



# Codes 
- Our codes version based on [BasicSR](https://github.com/xinntao/BasicSR). 


## Dataset

We use standard benchmark datasets: DF2K, Set5, Set14, BSD100, Urban100, and DIV2K100.
A complexity-annotated version of DF2K is provided, constructed through instance segmentation and complexity computation.

📂 Download (Baidu Netdisk):
https://pan.baidu.com/s/10rBTazmYvPdEsg-Khkh39w?pwd=kdzf

Access Code: kdzf

This dataset is for academic research only.

## Dataset Complexity Calculation

CASSRNet introduces a complexity-based data annotation mechanism, computed automatically via the following script:

D:\CNN\fuwuqishiyan\ClassSR-main\classSR_yolov8\codes\models\yolov8_complexities_log.py

## Function:

This script performs:

Instance segmentation using YOLOv8.

Per-region complexity scoring based on edge density, texture entropy, and gradient magnitude.

Classification of sub-images into simple, medium, and complex levels.

Logging of complexity distribution statistics for dataset balancing.

# Usage:
cd codes/models
python yolov8_complexities_log.py

# Output:

A .log file recording the computed complexity values for each sub-image.

Annotated dataset folders organized as:

./datasets/DF2K_complexity/simple/
./datasets/DF2K_complexity/medium/
./datasets/DF2K_complexity/complex/


This complexity information is later used during adaptive training and dynamic inference routing.



## How to Test CASSRNet

Clone the repository:

git clone https://github.com/FANRUIY/CASSRNet.git
cd CASSRNet

## Download the datasets and pretrained models, place them under:

./experiments/pretrained_models/


## Run testing:

cd codes
python test_CASSRNet.py -opt options/test/test_CASSRNet.yml


## Output results will be saved in:

./results/

## Option 2 — Run the Full Dynamic Inference Strategy Script

To reproduce the complexity-aware dynamic inference mechanism, run the provided test file:

D:~codes\FF_test_ClassSR.py


or, if using relative path (after entering your project folder):

python codes/FF_test_ClassSR.py


This script executes the adaptive branch selection based on instance complexity and evaluates runtime performance (FLOPs reduction, PSNR, SSIM).

## How to Train CASSRNet

Prepare training data:

cd codes/data_scripts
python data_augmentation.py
python generate_mod_LR_bic.py
python extract_subimages_train.py
python divide_subimages_train.py


## Start training:

cd codes
python train_CASSRNet.py -opt options/train/train_CASSRNet.yml


Trained models and logs are stored in:

./experiments/


# Model Overview

DCSAB — Depthwise-Separable Convolutional Residual Attention Block

HC-SA — Hybrid Channel–Spatial Attention

WA — Window Attention

PBS — Pyramid Branch Structure

Dynamic Selector — Complexity-driven branch routing

## Citation

If you find this work useful, please cite:

@article{fan2025cassrnet,
  title={Enhancing Image Super-Resolution via Complexity-Aware and Dynamic Inference Networks},
  author={Fan, Ruiyang and Sun, Guobing},
  journal={Applied Intelligence},
  year={2025}
}

## Acknowledgements

This project builds upon the following open-source works:

RCAN (ECCV 2018)

ClassSR (CVPR 2021)

BasicSR Framework

