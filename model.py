import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 30)

        self.dropout1 = nn.Dropout(p=0.19)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.19)

    def forward(self, input):


        res = self.dropout1(input)
        res = self.dropout2(F.leaky_relu(self.fc1(res)))
        res = self.dropout3(F.leaky_relu(self.fc2(res)))
        res = self.dropout4(F.leaky_relu(self.fc3(res)))
        res = self.fc4(res)

        # res = F.relu(self.fc1(input))
        # res = F.relu(self.fc2(res))
        # res = F.relu(self.fc3(res))
        # res = self.fc4(res)
        # res = self.fc4(res)
        # print("res_shape_is:", res.shape)
        #res = res.view(-1, 1)
        return res