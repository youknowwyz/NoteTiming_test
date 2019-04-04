import re
import collections
import os
import numpy as np
import phenome_classify as pc
import sub_string as sb
import data_process
import pre_process
import calc_target as ct

#从文件中按列读取数据
root_mono = "labels/mono"
root_full = "labels/full"
file_list_mono = sb.traverse_dir(root_mono)
file_list_full = sb.traverse_dir(root_full)
sb.read_files(file_list_full)
sb.read_files_time(file_list_full)
sb.read_mono(file_list_mono)



#####从按列保存的数据中读取所需行(每个note保存音节核)信息
data_process.read_data()

#####
dir = "res/note_lines.npy"
dir_time = "res/note_time.npy"
dir_mono = "res/note_mono_lines.npy"

##### 从按音节保存的行信息中读取所需要的特征，并保存到到all_train.npy
pre_process.get_train_data(dir, dir_time)

##### 从all_train.npy中按谱面时间和mono的时间计算target, 并保存target在[-15,14]的行
##### 最后的shape是target和data的shape 按照这个改模型输入的神经元
ct.get_targets("res/note_time.npy", "res/note_mono_lines.npy", "res/all_train.npy")