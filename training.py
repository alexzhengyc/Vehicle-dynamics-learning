import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import MLP2
from model import MLP2plus
from model import LSTM
from Datasets.datasets import zonda_Dataset
from Datasets.datasets import MLP_Dataset
from Datasets.datasets import RNN_Dataset

read_dir = 'Tire1-data1'
# write_dir = 'Tire1-data1/5/LSTM2'
write_dir = 'Tire1-data1/5/LSTM1'

# os.mkdir('Model/' + write_dir)

n_state = 5
n_control = 5

writer8_0 = SummaryWriter('runs/' + write_dir + '/8_0')
writer8_1 = SummaryWriter('runs/' + write_dir + '/8_1')
writer8_2 = SummaryWriter('runs/' + write_dir + '/8_2')
writer8_3 = SummaryWriter('runs/' + write_dir + '/8_3')
writer16_0 = SummaryWriter('runs/' + write_dir + '/16_0')
writer16_1 = SummaryWriter('runs/' + write_dir + '/16_1')
writer16_2 = SummaryWriter('runs/' + write_dir + '/16_2')
writer16_3 = SummaryWriter('runs/' + write_dir + '/16_3')
writer32_0 = SummaryWriter('runs/' + write_dir + '/32_0')
writer32_1 = SummaryWriter('runs/' + write_dir + '/32_1')
writer32_2 = SummaryWriter('runs/' + write_dir + '/32_2')
writer32_3 = SummaryWriter('runs/' + write_dir + '/32_3')
writer64_0 = SummaryWriter('runs/' + write_dir + '/64_0')
writer64_1 = SummaryWriter('runs/' + write_dir + '/64_1')
writer64_2 = SummaryWriter('runs/' + write_dir + '/64_2')
writer64_3 = SummaryWriter('runs/' + write_dir + '/64_3')

writer_graph = SummaryWriter('runs/' + write_dir + '/graph')


def writer_choose(neurons, lag):
    if neurons == 8:
        if lag == 0:
            w = writer8_0
        elif lag == 1:
            w = writer8_1
        elif lag == 2:
            w = writer8_2
        elif lag == 3:
            w = writer8_3

    elif neurons == 16:
        if lag == 0:
            w = writer16_0
        elif lag == 1:
            w = writer16_1
        elif lag == 2:
            w = writer16_2
        elif lag == 3:
            w = writer16_3

    elif neurons == 32:
        if lag == 0:
            w = writer32_0
        elif lag == 1:
            w = writer32_1
        elif lag == 2:
            w = writer32_2
        elif lag == 3:
            w = writer32_3

    elif neurons == 64:
        if lag == 0:
            w = writer64_0
        elif lag == 1:
            w = writer64_1
        elif lag == 2:
            w = writer64_2
        elif lag == 3:
            w = writer64_3

    else:
        w = -1
    return w


def normalize(data, dir):

    max_value = data.max(axis=0)
    min_value = data.min(axis=0)
    norm_np = np.vstack((max_value, min_value))
    df_norm = pd.DataFrame(norm_np, columns=["vx", "vy", "omega", "throttle", "steering", "dvx", "dvy", "dr"])
    df_norm.to_csv("Data/" + dir + "/normalize.csv")

    for i in range(data.shape[0]):
        data[i] = (data[i] - min_value) / (max_value - min_value)

    df_data = pd.DataFrame(data, columns=["vx", "vy", "omega", "throttle", "steering", "dvx", "dvy", "dr"])
    df_data.to_csv("Data/" + dir + "/data_norm.csv")

    return data


def denormalize(norm, dir):
    df = pd.read_csv("Data/" + dir + "normalize.csv")
    print(df.shape)
    max_value = df.iloc[0, :3]
    min_value = df.iloc[1, :3]
    norm = torch.detach(norm).numpy()
    denorm = np.dot(norm, max_value - min_value) + min_value
    return denorm


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer):
    size = int(len(dataloader.dataset) / dataloader.batch_size)
    print("training data size:", size)
    for batch, (x, y) in enumerate(dataloader):

        x = x.float()
        y = y.float()

        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        for i in range(1, y_pred.shape[1]):
            loss += loss_fn(y_pred[:, i], y[:, i])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        writer.add_scalar(f"Loss/train/{epoch}", loss, batch)


def valid_loop(dataloader, model, loss_fn, epoch, writer):
    size = len(dataloader.dataset)
    print("valid data size:", size)
    num_batches = len(dataloader)
    valid_loss, valid_error = 0, 0

    with torch.no_grad():
        for cut_x, cut_y in dataloader:

            cut_x = cut_x.float()
            cut_y = cut_y.float()

            y_pred = model(cut_x)

            for i in range(0, y_pred.shape[1]):
                loss = loss_fn(y_pred[:, i], cut_y[:, i])
                error = np.linalg.norm(y_pred[0, i] - cut_y[0, i])
                # error = np.linalg.norm(y_pred[0, i].cpu().numpy() - cut_y[0, i].cpu().numpy())
                valid_loss += loss.item()
                valid_error += error

    valid_loss /= num_batches
    valid_error /= num_batches

    writer.add_scalar(f"Loss/valid", valid_loss, epoch)
    writer.add_scalar(f"Error/valid", valid_error, epoch)

    print(f"Valid Error: \n Mn error {valid_error:>0.6f}, Avg loss: {valid_loss:>8f} \n")
    return valid_loss, valid_error


# Press the green button in the gutter to run the script.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    df = pd.read_csv("Data/" + read_dir + "/data.csv")
    data = df.values[:, 1:]
    data = data.astype("float")
    print("data type:", df.dtypes)
    data = normalize(data, read_dir)

    # print(torch.cuda.is_available())

    N = data.shape[0]

    learning_rate = 1e-4
    batch_size = 8
    epochs = 200

    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()

    for neurons in [16, 32, 64]:
        for lag in [0, 1, 2, 3]:

            # training_data = MLP_Dataset(data[:int(N * 0.8)], steps=1, n_state=n_state, n_control=n_control, lag=lag)
            # valid_data = MLP_Dataset(data[int(-N * 0.2):int(-N * 0.1)], steps=1, n_state=n_state, n_control=n_control,
            #                          lag=lag)
            # test_data = MLP_Dataset(data[int(-N * 0.1):], steps=1, n_state=n_state, n_control=n_control, lag=lag)
            training_data = RNN_Dataset(data[:int(N * 0.8)], steps=1, n_state=n_state, lag=lag)
            valid_data = RNN_Dataset(data[int(-N * 0.2):int(-N * 0.1)], steps=1, n_state=n_state, lag=lag)
            test_data = RNN_Dataset(data[int(-N * 0.1):], steps=1, n_state=n_state, lag=lag)

            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

            t1 = time.time()
            writer = writer_choose(neurons, lag)
            if writer == -1:
                break

            # *******************************************************************************************************
            # choose model
            # *******************************************************************************************************
            # model = MLP2(neurons, steps=1, n_state=n_state, n_control=n_control)
            # model = MLP2plus(neurons, steps=1, n_state=n_state, n_control=n_control)
            model = LSTM(hidden_dim=neurons, num_layers=1)

            total = sum([param.nelement() for param in model.parameters()])
            print("MODEL Parameters:", total)

            # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.8)
            min_loss = 1

            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer, t, writer)
                valid_loss, valid_error = valid_loop(valid_dataloader, model, loss_fn, t, writer)
                if valid_loss < min_loss:
                    min_loss = valid_loss
                scheduler.step()

            t2 = time.time()
            print(f"Total time: {t2 - t1}")
            print("Done!\n")

            writer_graph.add_scalar(f"Min loss", min_loss, neurons)
            writer_graph.add_scalar(f"Training Time", t2-t1, neurons)

            # torch.save(model.state_dict(), 'Model/' + directory + str(neurons) + '.pth')
            torch.save(model, "Model/" + write_dir + "_" + str(neurons) + "_" + str(lag) + ".pth")





