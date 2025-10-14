import sys
import os

# 将项目根目录加入 sys.path，方便导入 models 包
sys.path.append('/home/ruiyang/FRY/classSR_yolov8/codes')
import torch
from thop import profile
from models.archs.RCAN_arch import RCAN

# 输入尺寸
input_size = (1, 3, 64, 64)  # [Batch, C, H, W]

# 定义 3 个分支
rcan1 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=36, res_scale=1, n_colors=3, rgb_range=1, scale=4, reduction=16)
rcan2 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=50, res_scale=1, n_colors=3, rgb_range=1, scale=4, reduction=16)
rcan3 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1, n_colors=3, rgb_range=1, scale=4, reduction=16)

# 输入张量
x = torch.randn(*input_size)

# 分支 1
flops1, params1 = profile(rcan1, inputs=(x,))
print(f"RCAN Branch 1: FLOPs={flops1/1e9:.4f} GFLOPs, Params={params1/1e6:.4f} M")

# 分支 2
flops2, params2 = profile(rcan2, inputs=(x,))
print(f"RCAN Branch 2: FLOPs={flops2/1e9:.4f} GFLOPs, Params={params2/1e6:.4f} M")

# 分支 3
flops3, params3 = profile(rcan3, inputs=(x,))
print(f"RCAN Branch 3: FLOPs={flops3/1e9:.4f} GFLOPs, Params={params3/1e6:.4f} M")
