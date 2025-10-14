#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils/complexity.py

图像复杂度计算工具
提供函数 calculate_complexity(mask) 用于计算单张灰度图像的复杂度。
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ---------------------------- 特征计算 ----------------------------

def edge_complexity(mask: np.ndarray) -> int:
    """
    计算图像边缘长度（Canny边缘）
    mask: 灰度图或二值图，范围[0,1]
    """
    edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
    return int(np.sum(edges))

def instance_area(mask: np.ndarray) -> int:
    """计算图像区域面积"""
    return int(np.sum(mask))

def texture_complexity(mask: np.ndarray, distances=[1], angles=[0]):
    """
    计算灰度共生矩阵纹理特征
    返回 contrast, homogeneity, energy, entropy
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    glcm = graycomatrix(mask_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    glcm_norm = glcm / (glcm.sum() + 1e-10)
    entropy = -float(np.sum(glcm_norm * np.log(glcm_norm + 1e-10)))
    return contrast, homogeneity, energy, entropy

def calculate_complexity(mask: np.ndarray):
    """
    计算图像综合复杂度
    mask: 灰度图，范围[0,1]
    返回: combined_score, complexity_ratio, area, edge_len, texture_score
    """
    edge_len = edge_complexity(mask)
    area = instance_area(mask)
    contrast, homogeneity, energy, entropy = texture_complexity(mask)
    complexity_ratio = float(edge_len / area) if area > 0 else 0.0
    texture_score = float(contrast + homogeneity + energy + entropy)
    combined_score = complexity_ratio + texture_score
    return combined_score, complexity_ratio, area, edge_len, texture_score
