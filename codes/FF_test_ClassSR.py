#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import logging
import argparse
import time
from collections import OrderedDict
import yaml

import torch
import numpy as np
import cv2

import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from models.complexity import calculate_complexity

# 安全导入 dist 模块
try:
    import torch.distributed as dist
except ImportError:
    class DummyDist:
        def __init__(self):
            self.is_initialized = lambda: False
            self.get_rank = lambda: 0
            self.get_world_size = lambda: 1
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    dist = DummyDist()

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(data, logger):
    """确保数据预处理到 [0,1] 范围"""
    for key in ['LQ', 'GT']:
        if key in data:
            tensor = data[key]
            if tensor.max() > 1.0:
                data[key] = tensor / 255.0
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to main options YAML file.')
    args = parser.parse_args()

    opt = load_yaml_config(args.opt)
    opt['model'] = 'sr'

    # 分布式配置
    dist_enabled = opt.get('dist', False)
    if dist_enabled:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    else:
        rank = 0

    results_root = opt['path']['results_root']
    if rank == 0:
        os.makedirs(results_root, exist_ok=True)
        util.setup_logger('base', results_root, 'test_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    branches_cfg = opt['branches']
    thresholds = opt.get('complexity_thresholds', {'low': 212.79, 'high': 317.18})
    scale = opt.get('scale', 4)
    crop_border = opt.get('crop_border') or scale

    # FLOPs 权重 (GFLOPs)
    FLOPs_weights = {'branch1': 9.13, 'branch2': 12.15, 'branch3': 16.86}

    # -------- 加载分支模型 --------
    def get_model(branch_key):
        cfg = branches_cfg[branch_key]
        if 'model' not in cfg or cfg['model'] is None:
            logger.info(f"Loading model for {branch_key}")
            branch_network_config = cfg['network_G']
            checkpoint_path = cfg['path'].get('pretrain_model_G', None)
            strict_load = cfg['path'].get('strict_load', True)

            model_opt = {
                'network_G': branch_network_config,
                'is_train': False,
                'model': 'sr',
                'gpu_ids': [0] if torch.cuda.is_available() else None,
                'dist': dist_enabled,
                'train': {},
                'path': {'pretrain_model_G': checkpoint_path, 'strict_load': strict_load}
            }

            model = create_model(model_opt)

            # 加载权重
            if checkpoint_path and osp.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state_dict = None
                for key in ['params', 'params_ema', 'model', 'state_dict', 'generator']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint

                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                model.netG.load_state_dict(state_dict, strict=strict_load)

            model.netG = model.netG.to(device)
            model.netG.eval()
            cfg['model'] = model
            logger.info(f"Loaded model for {branch_key}")

        return cfg['model']

    branch_models = {k: get_model(k) for k in branches_cfg.keys()}

    # ---------------- 测试集加载 ----------------
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt.setdefault('data_type', 'img')
        dataset_opt.setdefault('io_backend', {'type': 'disk'})
        dataset_opt.setdefault('n_channels', 3)
        dataset_opt.setdefault('batch_size', 1)
        dataset_opt.setdefault('num_worker', 1)
        dataset_opt.setdefault('use_flip', False)
        dataset_opt.setdefault('use_rot', False)
        dataset_opt.setdefault('color', 'RGB')
        dataset_opt.setdefault('phase', 'test')
        dataset_opt.setdefault('GT_size', 256)
        dataset_opt['scale'] = scale

        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        logger.info(f'Number of test images in [{dataset_opt["name"]}]: {len(test_set)}')

    # ---------------- 测试循环 ----------------
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        dataset_dir = osp.join(results_root, test_set_name)
        os.makedirs(dataset_dir, exist_ok=True)
        test_results = OrderedDict(psnr=[], ssim=[], psnr_y=[], ssim_y=[], inference_times=[])

        branch_counts = {'branch1':0, 'branch2':0, 'branch3':0}  # 分支数量统计

        for idx, data in enumerate(test_loader):
            need_GT = 'GT' in data
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            data = preprocess_data(data, logger)

            # 计算复杂度选择分支
            lq_img = util.tensor2img(data['LQ'][0])
            gray = cv2.cvtColor(lq_img, cv2.COLOR_BGR2GRAY) / 255.0
            try:
                score, _, _, _, _ = calculate_complexity(gray)
            except:
                score = 250
            if score < thresholds['low']:
                branch_key = 'branch1'
            elif score <= thresholds['high']:
                branch_key = 'branch2'
            else:
                branch_key = 'branch3'

            branch_counts[branch_key] += 1

            model = branch_models[branch_key]
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            model.netG.eval()
            with torch.no_grad():
                model.feed_data(data, need_GT=need_GT)
                start_time = time.time()
                model.test()
                inference_time = time.time() - start_time
                test_results['inference_times'].append(inference_time)
                visuals = model.get_current_visuals(need_GT=need_GT)

            sr_tensor = torch.clamp(visuals['rlt'], 0, 1)
            sr_img = util.tensor2img(sr_tensor, out_type=np.uint8, min_max=(0, 1))
            save_img_path = osp.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            # 直接按分支输出 FLOPs
            flops = FLOPs_weights[branch_key]
            percent = flops / max(FLOPs_weights.values())
            logger.info(f"{img_name} - complexity: {score:.2f}, selected branch: {branch_key}, "
                        f"FLOPs: {flops:.2f} GFLOPs, Percent: {percent:.6f}")

            # PSNR/SSIM
            if need_GT:
                gt_img = util.tensor2img(data['GT'][0], out_type=np.uint8, min_max=(0,1))
                if crop_border > 0:
                    sr_img, gt_img = util.crop_border([sr_img, gt_img], crop_border)
                psnr = util.calculate_psnr(sr_img, gt_img)
                ssim = util.calculate_ssim(sr_img, gt_img)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)

                if gt_img.shape[2]==3:
                    sr_img_y = bgr2ycbcr(sr_img/255., only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img/255., only_y=True)
                    psnr_y = util.calculate_psnr(sr_img_y*255, gt_img_y*255)
                    ssim_y = util.calculate_ssim(sr_img_y*255, gt_img_y*255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    logger.info(f"{img_name} - PSNR: {psnr:.6f}; SSIM: {ssim:.6f}; "
                                f"PSNR_Y: {psnr_y:.6f}; SSIM_Y: {ssim_y:.6f}")
                else:
                    logger.info(f"{img_name} - PSNR: {psnr:.6f}; SSIM: {ssim:.6f}")

        # -------- 全测试集 FLOPs/Percent --------
        total_images = sum(branch_counts.values())
        total_flops = (
            branch_counts['branch1'] * FLOPs_weights['branch1'] +
            branch_counts['branch2'] * FLOPs_weights['branch2'] +
            branch_counts['branch3'] * FLOPs_weights['branch3']
        ) / total_images
        total_percent = total_flops / max(FLOPs_weights.values())

        logger.info(f"# Validation # selected branch1: {branch_counts['branch1']}; "
                    f"selected branch2: {branch_counts['branch2']}; "
                    f"selected branch3: {branch_counts['branch3']}; all: {total_images}")
        logger.info(f"# FLOPs {total_flops:.2f} GFLOPs Percent {total_percent:.6f}")

        # 平均指标
        if test_results['psnr']:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            avg_time = sum(test_results['inference_times']) / len(test_results['inference_times'])
            logger.info(f'----Average Results for {test_set_name}----')
            logger.info(f'\tPSNR: {ave_psnr:.6f}')
            logger.info(f'\tSSIM: {ave_ssim:.6f}')
            logger.info(f'\tInference Time: {avg_time:.4f}s')
            if test_results['psnr_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info(f'\tPSNR_Y: {ave_psnr_y:.6f}; SSIM_Y: {ave_ssim_y:.6f}')

    if dist_enabled and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()


#python codes/FF_test_ClassSR.py -opt codes/options/test/configs/merged_branch.yml
#三分支分类超分