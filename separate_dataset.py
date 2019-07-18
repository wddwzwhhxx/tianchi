classify_map = {}
for line in open("classify.csv","r"):
    line = line.strip('\n')
    key = line.split(',')[0].zfill(5)
    value = line.split(',')[1]
    classify_map[key] = value

import os
import shutil
import glob

paths = glob.glob(os.path.join('/home/wangzhaowei/rongyf/data/mask_data','*'))
for path in paths:
    label = path.split('/')[-1]
    new_path = path.replace(label,classify_map[(label)]).replace('mask_data','seven_classes_mask')
    # print('cp -r '+path+' '+new_path)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    os.system('cp -r '+path+' '+new_path)
# os.system('cp -r /home/feng/Documents/data/raw /home/feng/Documents')
