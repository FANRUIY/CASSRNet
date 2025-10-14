
import os.path as osp
import os
import numpy as np
import shutil


#divide training data
LR_folder="/home/ruiyang/FRY/classSR_yolov8/datasets_seg/DIV2K/DIV2K_valid_LR_bicubic/X3"
GT_folder="/home/ruiyang/FRY/classSR_yolov8/datasets_seg/DIV2K/DIV2K_valid_HR"

save_list=["/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_LR_bicubic/X3/class3",
           "/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_LR_bicubic/X3/class2",
           "/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_LR_bicubic/X3/class1",
           "/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_HR/class3",
           "/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_HR/class2",
           "/home/ruiyang/FRY/classSR_yolov8/datasets_seg_divide/DIV2K_divide/DIV2K_valid_HR/class1"]
for i in save_list:
    if os.path.exists(i):
        pass
    else:
        os.makedirs(i)
threshold=[80.1927,146.7028]#DIV2K_valid_HR的阈值

#f1 = open("/data0/xtkong/ClassSR-github/codes/data_scripts/divide_val.log")
f1 = open("/home/ruiyang/FRY/classSR_yolov8/codes/data_scripts/complexity_simple_DIV2K_valid_HR.log")
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
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[2], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[5], filename))
        if complexity >= threshold[0] and complexity < threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[1], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[4], filename))
        if complexity >= threshold[1]:
            shutil.copy(osp.join(LR_folder, filename), osp.join(save_list[0], filename))
            shutil.copy(osp.join(GT_folder, filename), osp.join(save_list[3], filename))

f1.close()

#复杂度低于第一个阈值：Class 1
#介于两个阈值：Class 2
#高于第二个阈值：Class 3

#threshold=[301.2311,396.6328]#set5HR的阈值
#threshold=[90.1296,150.5354]#DF2K_HR的阈值
#threshold=[92.0793,152.6547]#DIV2K_train_HR的阈值
#threshold=[80.1927,146.7028]#DIV2K_valid_HR的阈值
