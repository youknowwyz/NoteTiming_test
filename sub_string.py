import re
import collections
import os
import numpy as np

# what we need
#     1 e8 用32分音符*3来计长度 即96分音符
#     2 d8 用32分音符*3来计长度 即96分音符
#     3 phenome_classify
#     4 e22 e23
#     5 phenome_classify
#     6
#     7 phenome_classify

pattern = r'[@|^|-|+|=|_|%|-|~|!|[|]|$|+|&|#|;|||]'

data = "c@n^o-y+u=u_00%00^00_00~00-1!2[xx$1]xx/A:2-1-1@JPN~0/B:2_1_1@JPN|0/C:1+1+1@JPN&0/D:F4!0#11$4/4%100|1&30;12-xx/E:G4]2^11=4/4~100!1@60#24+xx]4$2|12[12&48]48=50^50~4#4_12;29$48&120%28[72|0]0-mp^xx+xx~xx=xx@xx$xx!xx%xx#xx|xx|xx-xx&xx&xx+xx[xx;xx]xx;xx~xx~xx^xx^xx@xx[xx#xx=xx!xx~m2+p2!xx^xx/F:A4#4#11-4/4$100$1+60%24;xx/G:xx_xx/H:7_7/I:9_9/J:4~4@16"



root_mono = "labels/mono"
root_full = "labels/full"

def traverse_dir(root_dir, extension = '.lab'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))

    return file_list


######按字母分割整个串
def sub_data(data):
    data_list = []
    for char in range(65,75):
        ch = chr(char)
        data_tmp = data.split(str("/"+ch+":"))
        data_list.append(data_tmp[0])
        #print("/" + ch + ":" + data_tmp[0])
        data = str(data_tmp[1])
        if char == 74:
            #print("/" + ch + ":" + data_tmp[1])
            data_list.append(data_tmp[1])

    return data_list




def show_sub_data(data_list):
    for char in range(0,11):
        if char == 0:
            print("/P:" + data_list[0])
        else:
            print("/"+chr(char + 64)+":"+data_list[char])

#########把所有文件里的每一行按字母切成小段保存
def read_files(file_list):
    all_lines_list = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(' ')
                data = line_list[2]
                data_res = sub_data(data)
                show_sub_data(data_res)
                all_lines_list.append(data_res)

    all_lines_list = np.array(all_lines_list)
    print((all_lines_list).shape)
    np.save("res/all_lines.npy",all_lines_list)

#####保存mono里的数据到数组
def read_mono(file_list):
    all_mono_lines_list = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_tmp = []
                line_list = line.split(' ')
                line_tmp.append(line_list[0])
                line_tmp.append(line_list[1])
                line_tmp.append(line_list[2])
                all_mono_lines_list.append(line_tmp)

    all_mono_lines_list = np.array(all_mono_lines_list)
    print(all_mono_lines_list.shape)
    np.save("res/all_mono_lines.npy",all_mono_lines_list)


def see_max_length(dir):
    all_lines = np.load(dir)
    max = 0
    pattern = r'!|#|\$|%|\||&|;|-|\]|\^|=|~|@|\+'
    for i in range(all_lines.shape[0]):
        for j in range(4,7):
            str = all_lines[i][j]
            result = re.split(pattern, str)
            length = result[7]
            print(length)
            if length != "xx":
                num = int(length)
                if num > max:
                    max = num

    return max

########保存原始数据的time到数组
def read_files_time(file_list):
    all_time_list = []
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(' ')
                data_1 = line_list[0]
                data_2 = line_list[1]
                data_all = [data_1, data_2]
                all_time_list.append(data_all)
    all_time_list = np.array(all_time_list)
    print((all_time_list).shape)
    np.save("res/all_time.npy",all_time_list)

if __name__ == '__main__':
    file_list = traverse_dir(root_full)
    file_mono_list = traverse_dir(root_mono)
    read_files_time(file_list)
    read_files(file_list)
    read_mono(file_mono_list)


