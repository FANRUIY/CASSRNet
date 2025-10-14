#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像复杂度分析工具（流式处理版）
功能：计算分割图像的边缘复杂度、区域面积和纹理特征，自动拟合阈值，绘制分布图
输出：JSON格式、可读文本格式和简洁log格式的报告
"""

import os
import cv2
import numpy as np
import json
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------- 特征计算部分 ----------------------------

def edge_complexity(mask):
    edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
    return int(np.sum(edges))

def instance_area(mask):
    return int(np.sum(mask))

def texture_complexity(mask, distances=[1], angles=[0]):
    mask_uint8 = (mask * 255).astype(np.uint8)
    glcm = graycomatrix(mask_uint8, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    glcm_norm = glcm / (glcm.sum() + 1e-10)
    entropy = -float(np.sum(glcm_norm * np.log(glcm_norm + 1e-10)))
    return contrast, homogeneity, energy, entropy

def calculate_complexity(mask):
    edge_len = edge_complexity(mask)
    area = instance_area(mask)
    contrast, homogeneity, energy, entropy = texture_complexity(mask)
    complexity_ratio = float(edge_len / area) if area > 0 else 0.0
    texture_score = float(contrast + homogeneity + energy + entropy)
    combined_score = complexity_ratio + texture_score
    return combined_score, complexity_ratio, area, edge_len, texture_score

# ---------------------------- 流式处理部分 ----------------------------

def calculate_all_complexities_streaming(directory):
    complexities = {}
    all_scores = []

    for file in os.listdir(directory):
        if not file.lower().endswith('.png'):
            continue

        file_path = os.path.join(directory, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ 无法读取图像: {file_path}")
            continue

        mask = (img / 255.0).astype(np.float32)

        try:
            combined_score, complexity_ratio, area, edge_len, texture_score = calculate_complexity(mask)
            complexities[file] = {
                "combined_score": combined_score,
                "complexity_ratio": complexity_ratio,
                "area": area,
                "edge_length": edge_len,
                "texture_score": texture_score
            }
            all_scores.append(combined_score)
        except Exception as e:
            print(f"❌ 处理 {file} 时出错: {str(e)}")
            complexities[file] = None

    return complexities, all_scores

# ---------------------------- 阈值拟合 ----------------------------

def fit_thresholds(scores, low_percentile=33, high_percentile=66):
    low_th = np.percentile(scores, low_percentile)
    high_th = np.percentile(scores, high_percentile)
    return low_th, high_th

# ---------------------------- 可视化 ----------------------------

def plot_score_distribution(scores, low_th=None, high_th=None, bins=50, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, color='skyblue', edgecolor='black')
    if low_th is not None:
        plt.axvline(x=low_th, color='green', linestyle='--', label=f'Low Threshold: {low_th:.2f}')
    if high_th is not None:
        plt.axvline(x=high_th, color='red', linestyle='--', label=f'High Threshold: {high_th:.2f}')
    plt.xlabel("Complexity Score")
    plt.ylabel("Instance Count")
    plt.title("Instance Complexity Score Distribution")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📊 复杂度分布图保存至: {save_path}")
    else:
        plt.show()

# ---------------------------- 保存 ----------------------------

def save_complexities_to_json(complexities, output_file):
    def type_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: type_converter(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [type_converter(x) for x in obj]
        return obj

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(type_converter(complexities), f, indent=4, ensure_ascii=False)

def save_complexities_to_txt(complexities, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 图像复杂度分析报告 ===\n\n")
        for file, metrics in complexities.items():
            if not metrics:
                f.write(f"{file}: 计算失败\n\n")
                continue
            f.write(f"{file}:\n")
            f.write(f"  • 综合复杂度分数: {metrics['combined_score']:.4f}\n")
            f.write(f"  • 复杂度比值: {metrics['complexity_ratio']:.4f}\n")
            f.write(f"  • 区域面积: {metrics['area']} 像素\n")
            f.write(f"  • 边缘长度: {metrics['edge_length']} 像素\n")
            f.write(f"  • 纹理得分: {metrics['texture_score']:.4f}\n\n")

def save_simple_log(complexities, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for file, metrics in complexities.items():
            if not metrics:
                continue
            now = datetime.now().strftime("%y-%m-%d %H:%M:%S.%f")[:-3]
            line = f"{now} - INFO: {file} - complexity: {metrics['combined_score']:.6f}\n"
            f.write(line)

# ---------------------------- 主程序 ----------------------------

if __name__ == "__main__":
    INPUT_DIR = '/home/ruiyang/FRY/ultralytics/output/DIV2K_valid_HR'
    JSON_OUTPUT = '/home/ruiyang/FRY/ultralytics/output/metrics/complexities.json'
    TXT_OUTPUT = '/home/ruiyang/FRY/ultralytics/output/metrics/complexities.txt'
    SIMPLE_LOG_OUTPUT = '/home/ruiyang/FRY/ultralytics/output/metrics/complexity_simple.log'
    PLOT_OUTPUT = '/home/ruiyang/FRY/ultralytics/output/metrics/complexity_distribution.png'

    print("🟢 图像复杂度分析开始...")

    # 1. 流式处理计算
    print("➊ 正在流式处理计算图像...")
    complexities, all_scores = calculate_all_complexities_streaming(INPUT_DIR)
    if len(all_scores) == 0:
        print("❌ 错误：未计算到任何复杂度分数！")
        exit(1)

    # 2. 阈值拟合
    print("➋ 自动拟合复杂度阈值...")
    low_th, high_th = fit_thresholds(all_scores)
    print(f"✅ 拟合阈值：低阈值 = {low_th:.4f}, 高阈值 = {high_th:.4f}")

    # 3. 保存结果
    print("➌ 正在保存结果...")
    save_complexities_to_json(complexities, JSON_OUTPUT)
    save_complexities_to_txt(complexities, TXT_OUTPUT)
    save_simple_log(complexities, SIMPLE_LOG_OUTPUT)

    # 4. 可视化
    print("➍ 正在绘制复杂度分布图...")
    plot_score_distribution(all_scores, low_th=low_th, high_th=high_th, save_path=PLOT_OUTPUT)

    print(f"""\n✅ 分析完成！
JSON报告: {JSON_OUTPUT}
文本报告: {TXT_OUTPUT}
简单LOG: {SIMPLE_LOG_OUTPUT}
分布图: {PLOT_OUTPUT}
阈值: 低阈值={low_th:.4f}, 高阈值={high_th:.4f}
实例复杂度分数数量: {len(all_scores)}
""")
