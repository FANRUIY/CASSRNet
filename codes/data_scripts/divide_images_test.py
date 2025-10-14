
import os.path as osp
import os
import numpy as np
import shutil


#divide testing data for single SR models

LR_folder="/home/ruiyang/FRY/ClassSR/datasets/DIV2K/DIV2K_valid_LR_bicubic/X4_sub"
GT_folder="/home/ruiyang/FRY/ClassSR/datasets/DIV2K/DIV2K_valid_HR_sub"

save_list=["/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_LR_class3",
           "/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_LR_class2",
           "/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_LR_class1",
           "/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_GT_class3",
           "/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_GT_class2",
           "/home/ruiyang/FRY/ClassSR/datasets/DIV2K_divide/DIV2K_valid_HR_sub_psnr_GT_class1"]
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)
threshold=[27.16882,35.149761]

f1 = open("/home/ruiyang/FRY/ClassSR/codes/data_scripts/divide_val.log")
#f1 = open("/data0/xtkong/ClassSR-github/codes/data_scripts/divide_train.log")
a1 = f1.readlines()
index=0
for i in a1:
    index += 1
    print(index)
    if ('- INFO:' in i and '- complexity:' in i):
        # 提取复杂度数值
        complexity = float(i.split('- complexity: ')[1].strip())
        
        # 提取文件名
        filename = i.split('- INFO: ')[1].split(' - complexity:')[0].strip()
        
        print(filename, complexity)

        if complexity < threshold[0]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[0], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))
        if complexity >= threshold[0] and complexity < threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[1], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[4], filename))
        if complexity >= threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[2], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[5], filename))

f1.close()
