import re
import collections
import os
import numpy as np
from collections import Counter
root_dir = "labels/mono"

def traverse_dir(root_dir, extension = '.lab'):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))

    return file_list

def read_mono(file_list):
    all_mono_lines_list = []
    counts = [0,0,0]
    sil_index = []
    br_index = []
    cl_index = []
    for file_num, file in enumerate(file_list):
        sil = 0
        br = 0
        cl = 0
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_tmp = []
                line_list = line.split(' ')
                print(line_list[2])

                if line_list[2] == "sil\n":
                    sil = 1
                if line_list[2] == "br\n":
                    br = 1
                if line_list[2] == "cl\n":
                    cl = 1
        if sil == 1:
            counts[0] += 1
            sil_index.append(file_num)
        if br == 1:
            counts[1] += 1
            br_index.append(file_num)
        if cl == 1:
            counts[2] += 1
            cl_index.append(file_num)


    print(counts)
    print(sil_index)
    print(br_index)
    print(cl_index)


if __name__ == '__main__':

    data_line = np.load("res/test/test_data.npy")
    print(data_line)
    # file_list = traverse_dir(root_dir)
    # read_mono(file_list)
    # pattern = r'!|#|\$|%|\||&|;|-|\]|\^|=|~|@|\+|\[|\_'
    # all_lines = np.load("res/all_lines.npy")
    # num = 0
    # for i in range(all_lines.shape[0]):
    #     str = all_lines[i][5]
    #     p = all_lines[i][0]
    #     result = re.split(pattern, str)
    #     ps = re.split(pattern, p)
    #     if result[5] != "1":
    #         num+=1
    #         print(i, ps[3], result[5])
    #
    # print(num)
    # #
