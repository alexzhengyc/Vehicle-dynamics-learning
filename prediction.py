import time

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from Datasets.datasets import MLP_Dataset
from Datasets.datasets import RNN_Dataset
from torch.utils.tensorboard import SummaryWriter

directory = 'TinyCar2%-1/'
# model_type = "MLP2plus"
model_type = "LSTM1"
writer1 = SummaryWriter('eval/' + directory + model_type + '/one-step/normalized ground truth')
writer2 = SummaryWriter('eval/' + directory + model_type + '/one-step/normalized prediction')
writer3 = SummaryWriter('eval/' + directory + model_type + '/one-step/ground truth')
writer4 = SummaryWriter('eval/' + directory + model_type + '/one-step/prediction')
writer5 = SummaryWriter('eval/' + directory + model_type + '/10-steps/normalized ground truth')
writer6 = SummaryWriter('eval/' + directory + model_type + '/10-steps/normalized prediction')
writer7 = SummaryWriter('eval/' + directory + model_type + '/10-steps/ground truth')
writer8 = SummaryWriter('eval/' + directory + model_type + '/10-steps/prediction')

RMSE1_w = SummaryWriter('eval/' + directory + model_type + '/one-step/RMSE')
error1_vx_w = SummaryWriter('eval/' + directory + model_type + '/one-step/vx_error')
error1_vy_w = SummaryWriter('eval/' + directory + model_type + '/one-step/vy_error')
error1_r_w = SummaryWriter('eval/' + directory + model_type + '/one-step/r_error')

n_state = 5
n_control = 5
lag = 0
neurons = 32
Ts = 0.02


def update(x_prev, x, y):                    # x_prev:normalized tensor x:normalized tensor y: normalized numpy

    x_prev = x_prev.detach().numpy().squeeze(0)
    x = x.detach().numpy().squeeze(0)
    # print(f"x shape: {x.shape}")    # (1, 5, 5)

    x_old = x_prev[1:, :]

    state_new = denormalize(x_prev[-1, :3], type=1) + denormalize(y, type=2)
    control_new = x[-1:, -2:]
    x_new = np.concatenate((state_new, control_new), axis=1)
    # print(f"x new shape: {x_new.shape}")

    x = np.concatenate((x_old, x_new), axis=0)
    # print(f"updated x shape: {x.shape}")

    x = normalize(x, type=0)
    x = np.expand_dims(x, axis=0)
    # print(f"updated x shape: {x.shape}")
    x = torch.tensor(x, requires_grad=True)
    return x


def normalize(data, type):             # data: 2D numpy

    df = pd.read_csv("Data/" + directory + "normalize.csv")
    norm = df.values[:, 1:]
    if type == 0:                      # normalize MLP input data
        max_value = norm[0, :5]
        min_value = norm[1, :5]
        data = data.reshape(-1, 5)

    elif type == 1:                    # normalize state data
        max_value = norm[0, :3]
        min_value = norm[1, :3]
        data = data.reshape(-1, 3)
    elif type == 2:                    # normalize output data
        max_value = norm[0, -3:]
        min_value = norm[1, -3:]
        data = data.reshape(-1, 3)
    else:
        return -1

    for i in range(data.shape[0]):
        data[i] = (data[i] - min_value) / (max_value - min_value)
    # data = data.reshape(1, -1)

    return data


def denormalize(data, type):           # data: 2D numpy
    df = pd.read_csv("Data/" + directory + "normalize.csv")
    norm = df.values

    if type == 0:                      # normalize input data
        max_value = norm[0, :5]
        min_value = norm[1, :5]
        data = data.reshape(-1, 5)
    elif type == 1:                    # normalize state data
        max_value = norm[0, :3]
        min_value = norm[1, :3]
        data = data.reshape(-1, 3)
    elif type == 2:                    # normalize output data
        max_value = norm[0, -3:]
        min_value = norm[1, -3:]
        data = data.reshape(-1, 3)
    else:
        return -1
    # print(f"max value shape {max_value.shape}")

    for i in range(data.shape[0]):
        data[i] = data[i] * (max_value - min_value) + min_value
    # data = data.reshape(1, -1)
    return data


def one_step_loop(dataloader, model):

    # ***********************************************************************************************
    # one-step prediction error
    # ***********************************************
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):

            x = x.float()
            y = y.float()

            y_pred = model(x)
            y = y.detach().numpy().squeeze(0)
            y_pred = y_pred.detach().numpy().squeeze(0)
            # ************************************************************************************
            # RMSE error = linalg.norm / sqrt(n)
            error = np.linalg.norm(y_pred - y) / np.sqrt(3)
            RMSE1_w.add_scalar("RMSE", error, batch)

            y_true = denormalize(y, type=2)
            # print(f"y_ture shape {y_true.shape}")
            y_pred_true = denormalize(y_pred, type=2)
            error1_vx_w.add_scalar("error of vx", np.abs(y_true[0, 0] - y_pred_true[0, 0]), batch)
            error1_vy_w.add_scalar("error of vy", np.abs(y_true[0, 1] - y_pred_true[0, 1]), batch)
            error1_r_w.add_scalar("error of r", np.abs(y_true[0, 2] - y_pred_true[0, 2]), batch)

            if 10 <= batch < 510:
                # print(cut_x[0])
                writer1.add_scalar("dvx", y[0, 0], batch)
                writer2.add_scalar("dvx", y_pred[0, 0], batch)
                writer1.add_scalar("dvy", y[0, 1], batch)
                writer2.add_scalar("dvy", y_pred[0, 1], batch)
                writer1.add_scalar("dr", y[0, 2], batch)
                writer2.add_scalar("dr", y_pred[0, 2], batch)

                writer3.add_scalar("dvx_true", y_true[0, 0], batch)
                writer4.add_scalar("dvx_true", y_pred_true[0, 0], batch)
                writer3.add_scalar("dvy_true", y_true[0, 1], batch)
                writer4.add_scalar("dvy_true", y_pred_true[0, 1], batch)
                writer3.add_scalar("dr_true", y_true[0, 2], batch)
                writer4.add_scalar("dr_true", y_pred_true[0, 2], batch)

            elif batch > 510:
                t1 = time.time_ns()
                y_pred = model(x)
                t2 = time.time_ns()
                print(f"prediction time: {(t2 - t1) / 1e+6} ms")
                break


def multi_step_loop(dataloader, model):
    # ************************************************************************************************************
    # multi-step prediction error
    # ******************************************************
    steps = 10
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x = x.float()
            y = y.float()
            y = y.detach().numpy().squeeze(0)

            if batch in [10, 20, 30, 40, 50]:

                for i in range(steps):
                    y_pred = model(x_prev)
                    y_pred = y_pred.detach().numpy().squeeze(0)

                    writer5.add_scalar("dvx", y[0, 0], batch)
                    writer6.add_scalar("dvx", y_pred[0, 0], batch)
                    writer5.add_scalar("dvy", y[0, 1], batch)
                    writer6.add_scalar("dvy", y_pred[0, 1], batch)
                    writer5.add_scalar("dr", y[0, 2], batch)
                    writer6.add_scalar("dr", y_pred[0, 2], batch)

                    y_true = denormalize(y, type=2)
                    y_pred_true = denormalize(y_pred, type=2)
                    writer7.add_scalar("dvx_true", y_true[0, 0], batch)
                    writer8.add_scalar("dvx_true", y_pred_true[0, 0], batch)
                    writer7.add_scalar("dvy_true", y_true[0, 1], batch)
                    writer8.add_scalar("dvy_true", y_pred_true[0, 1], batch)
                    writer7.add_scalar("dr_true", y_true[0, 2], batch)
                    writer8.add_scalar("dr_true", y_pred_true[0, 2], batch)

                    x_prev = update(x_prev, x, y_pred)

            elif batch > 510:
                break

            x_prev = x
            y_prev = y


# Press the green button in the gutter to run the script.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # *************************************************************************************************************
    # data read
    # **************************************************
    df_norm = pd.read_csv("Data/" + directory + "data_norm.csv")
    df_true = pd.read_csv("Data/" + directory + "data.csv")

    data_norm = df_norm.values[:, 1:]
    data_true = df_true.values[:, 1:]
    N = data_norm.shape[0]
    # *************************************************************************************************************
    # dataloader
    # **************************************************
    # test_data = MLP_Dataset(data_norm[int(-N * 0.1):], steps=1, n_control=n_control, n_state=n_state, lag=lag)
    test_data = RNN_Dataset(data_norm[int(-N * 0.1):], steps=1, n_state=5, lag=lag)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    # *************************************************************************************************************
    # model
    # ***************************************************
    # model = torch.load("Model/" + directory + str(n_control) + "/" +
    #                    model_type + "_" + str(neurons) + "_" + str(lag) + ".pth")
    model = torch.load("Model/" + directory + str(n_state) + "/" +
                       model_type + "_" + str(neurons) + ".pth")
    one_step_loop(test_dataloader, model)
    multi_step_loop(test_dataloader, model)









