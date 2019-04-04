import re
import collections
import os
import numpy as np
import phenome_classify as pc
import random
from collections import Counter

def calc_target(time1, time2, flag):
    tmp = time2 - time1
    # 15 is 0(mid)
    #print(tmp)
    target = 0
    randnum = random.randint(-15,14)
    if flag == True:
        loc = int(tmp / 50000)
        # if loc > 14:
        #     loc = 14
        # if loc < -15:
        #     loc = -15

    else:
        loc = int(tmp / 100000)
        # if loc > 14:
        #     loc = 14
        # if loc < -15:
        #     loc = -15

    target = loc + 15
    if target > 50:
        print(time2, time1, target)
    return target

def get_targets(time_file, mono_file, train_file):
    time_lines = np.load(time_file)
    mono_lines = np.load(mono_file)
    all_train = np.load(train_file)
    print(time_lines.shape)
    print(mono_lines.shape)
    cleansed_train = []

    line_len = time_lines.shape[0]

    count_tar = Counter()

    targets_lines = []

    for i in range(line_len):
        time1 = float(time_lines[i][0])
        time2 = float(mono_lines[i][0])
        if pc.phonome_is_rest(mono_lines[i][2]) == 0:
            targets = calc_target(time1, time2, True)
        else:
            targets = calc_target(time1, time2, False)
        if targets > 50:
            print(i, time1, time2, targets)
        if targets >= 0 and targets<=29:
            count_tar[targets] += 1
            targets_lines.append(targets)
            cleansed_train.append(all_train[i])

    for key in count_tar.keys():
        print(key , ":" , count_tar[key])
    targets_lines = np.array(targets_lines)
    cleansed_train = np.array(cleansed_train)

    print(targets_lines.shape)
    print(cleansed_train.shape)
    for i in range(targets_lines.shape[0]):
        if targets_lines[i]>=30 or targets_lines[i]<0:
            print(targets_lines[i])

    val_target = targets_lines[-500:-1]
    test_target = targets_lines[-1000:-501]
    trainning_target = targets_lines[:-1001]
    val_data  = cleansed_train[-500: -1]
    test_data = cleansed_train[-1000: -501]
    trainning_data = cleansed_train[: -1001]


    #
    # for key in count_tar.keys():
    #     print(key, ":", count_tar[key])
    np.save("res/val_target.npy",val_target)
    np.save("res/test_target.npy", test_target)
    np.save("res/trainning_target.npy", trainning_target)
    np.save("res/val_data.npy", val_data)
    np.save("res/test_data.npy", test_data)
    np.save("res/trainning_data.npy", trainning_data)


if __name__ == '__main__':
    #calc_one_hot(24000000 ,23800000, True)
    get_targets("res/note_time.npy", "res/note_mono_lines.npy", "res/all_train.npy")