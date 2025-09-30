# CASSRNet
[Official] PyTorch Implementation of "CASSRNet: Image Super-Resolution Based on Complexity-Awareness and Dynamic Inference"
Dataset Preparation
We trained our models on the DF2K dataset.

Please download the dataset from [Official Website/Dataset Homepage].

Place the data in the data/ directory following the structure below。


We train the proposed model on the DF2K dataset (a combination of DIV2K and Flickr2K, consisting of 3,450 high-quality images) and evaluate it on five widely used benchmark datasets: Set5, Set14, BSD100, Urban100, and DIV2K100.
Different from conventional whole-image training, our method adopts an instance-level training and inference paradigm, where semantic segmentation is applied to extract foreground regions as inputs during both training and testing. This design enables the model to learn from finer-grained target regions, avoids interference from irrelevant background areas, and facilitates both complexity estimation and multi-branch training. At the same time, it enhances the network’s ability to model structural details and texture features.


Data Generation Process
If you wish to reproduce our data processing pipeline:
Start with original DF2K: Download from DIV2K Official and Flickr2K
Run instance segmentation: Use the provided script:
bash
python scripts/instance_segmentation.py --input_dir path/to/original/DF2K --output_dir data/DF2K_Processed/HR

Generate LR images:
bash
python scripts/prepare_data.py --hr_path data/DF2K_Processed/HR --lr_path data/DF2K_Processed/LR_bicubic/X4 --scale 4
Note: The instance segmentation model weights and configuration are provided in the segmentation/ directory.

