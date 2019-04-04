import torch
import os
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, name_d, name_t):

        self.data_X = np.load(os.path.join(path ,name_d))
        self.targets = np.load(os.path.join(path ,name_t))


        pass

    def __getitem__(self, index):
        #print(self.targets[index])
        return (self.data_X[index].astype(float), self.targets[index].astype(np.long))

    def __len__(self):
        return len(self.targets)

def get_my_data(BATCH_SIZE=32):

    # training_data = np.load("res/tra/trainning_data.npy")
    # training_target = np.load("res/tra/trainning_target.npy")
    # train_iter = torch.utils.data.DataLoader(
    #     [training_data, training_target],
    #     batch_size=BATCH_SIZE, shuffle=True,num_workers=10
    # )

    t_path = "res/tra"
    v_path = "res/val"
    transform = transforms.ToTensor()
    train_iter = torch.utils.data.DataLoader(
        MyDataset(t_path, "trainning_data.npy", "trainning_target.npy"),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
    )

    val_iter = torch.utils.data.DataLoader(
        MyDataset(t_path, "trainning_data.npy", "trainning_target.npy"),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=10,

    )

    return train_iter, val_iter


def get_test_data(BATCH_SIZE=32):
    te_path = "res/test"
    test_iter = torch.utils.data.DataLoader(
        MyDataset(te_path, "test_data.npy", "test_target.npy"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
    )

    return test_iter

if __name__ == '__main__':
    get_my_data()
    get_test_data()