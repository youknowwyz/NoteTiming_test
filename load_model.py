import torch
from torch import nn
import re
import collections
import os
import numpy as np
import dataset
import model
from torch import nn, optim

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_test = model.Model()
    model_test.double()
    state_dict = torch.load('checkpoints/checkpoint7.0.pth')
    print(state_dict.keys())
    model_test.load_state_dict(state_dict)
    print(model_test)

    test_iter = dataset.get_test_data()
    len_test = len(test_iter)

    Error = 0
    model_test.eval()
    with torch.no_grad():
        for batch_num, batch in enumerate(test_iter):
            data = batch[0]
            target = batch[1]
            target = target.long()

            ps = model_test(data)

            top_p, top_class = ps.topk(1, dim=1)
            print("predict:", top_class.view(-1))
            print("target:", target)


            equals = torch.abs(top_class - target)

            Error += torch.mean(equals.type(torch.FloatTensor))

    print("Accuracy of Test is:", Error/len_test)
