import sys
import os

# 加入项目根目录
sys.path.append('/home/ruiyang/FRY/classSR_yolov8/codes')

import torch
from thop import profile
from models.archs.RCAN_arch import RCAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 初始化 3 个 RCAN 子模型
net1 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=36, res_scale=1,scale=4).to(device)
net2 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=48, res_scale=1,scale=4).to(device)
net3 = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1,scale=4).to(device)

# 输入分辨率
input_tensor = torch.randn(1, 3, 64, 64).to(device)

def measure_flops_params(net, input_tensor):
    flops, params = profile(net, inputs=(input_tensor,), verbose=False)
    params_m = params / 1e6
    return flops, params_m  # 返回 FLOPs（乘加次数的总操作数），参数数目（M）

flops1, params1 = measure_flops_params(net1, input_tensor)
flops2, params2 = measure_flops_params(net2, input_tensor)
flops3, params3 = measure_flops_params(net3, input_tensor)

# FLOPs 转 MACs，1 MAC = 2 FLOPs
macs1 = flops1 / 2 / 1e9
macs2 = flops2 / 2 / 1e9
macs3 = flops3 / 2 / 1e9

print(f"Net1 (n_feats=36) FLOPs: {macs1:.2f} GMac, Params: {params1:.2f} M")
print(f"Net2 (n_feats=48) FLOPs: {macs2:.2f} GMac, Params: {params2:.2f} M")
print(f"Net3 (n_feats=64) FLOPs: {macs3:.2f} GMac, Params: {params3:.2f} M")

# 按类别权重求加权平均 FLOPs（用 MACs）
num_ress = [1, 1, 1]
flops_weighted_mac = (macs1 * num_ress[0] + macs2 * num_ress[1] + macs3 * num_ress[2]) / sum(num_ress)

print(f"Weighted FLOPs: {flops_weighted_mac:.2f} GMac")
