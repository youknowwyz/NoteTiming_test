import torch
from torch import nn
import re
import collections
import os
import numpy as np
import dataset
import model
from torch import nn, optim

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_test = model.Model()
    model_test.double()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_test.parameters(), lr=0.00007)

    epochs = 2800

    train_iter, val_iter = dataset.get_my_data()
    train_losses = []
    test_losses = []
    len_data = len(train_iter)
    len_val = len(val_iter)
    step = 0
    print_step = 20
    print("len of train is：", len_data)
    print("len of val is:", len_val)


    # model_test.to(device)
    for e in range(epochs):
        running_loss = 0
        for batch_num ,batch in enumerate(train_iter):
            #print("batch_num:", batch_num)
            #print("batch:", batch[0])
            step += 1
            #for batch in train_iter:

            data = batch[0]
            target = batch[1]
            target = target.long()

            optimizer.zero_grad()
            log_ps = model_test(data)
            loss = criterion(log_ps, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        #print(running_loss)
        else:
            val_loss = 0
            Accuracy0 = 0
            Accuracy2 = 0
            Accuracy4 = 0
            Error = 0
            model_test.eval()
            print("########################################################################")
            with torch.no_grad():
                for batch_num_v, batch in enumerate(val_iter):

                    data = batch[0]
                    target = batch[1]
                    target = target.long()

                    log_ps = model_test(data)
                    val_loss += criterion(log_ps, target)

                    # ps = torch.exp(log_ps)
                    #print(ps)
                    top_p, top_class = log_ps.topk(1, dim=1)
                    #print(top_class)
                    #print(top_p)
                    # print(top_p)
                    #print(top_class)
                    #print(target)
                    # equals = top_class == target.view(*top_class.shape)
                    # accuracy += torch.mean(equals.type(torch.FloatTensor))
                    predict = top_class.view(-1)
                    equals = torch.abs(predict - target)
                    # print(equals)
                    equals0 = torch.zeros(32)
                    equals2 = torch.zeros(32)
                    equals4 = torch.zeros(32)
                    for i in range(len(equals)):
                        if equals[i] <= 4:
                            equals4[i] = 1
                        else:
                            equals4[i] = 0

                        if equals[i] <= 2:
                            equals2[i] = 1
                        else:
                            equals2[i] = 0

                        if equals[i] == 0:
                            equals0[i] = 1
                        else:
                            equals0[i] = 0

                    Accuracy0 += torch.mean(equals0.type(torch.FloatTensor))
                    Accuracy2 += torch.mean(equals2.type(torch.FloatTensor))
                    Accuracy4 += torch.mean(equals4.type(torch.FloatTensor))


            model_test.train()

            train_losses.append(running_loss / len_data)
            test_losses.append(val_loss / len_data)

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len_data),
                  "Val Loss: {:.3f}.. ".format(val_loss / len_val),
                  "Acc0: {:.3f}".format(Accuracy0 / len_val),
                  "Acc2: {:.3f}".format(Accuracy2 / len_val),
                  "Acc4: {:.3f}".format(Accuracy4 / len_val))

            running_loss = 0


        if e % 140 == 0 and e > 0:
            torch.save(model_test.state_dict(), 'checkpoints/checkpoint'+ str(e/140) + '.pth')