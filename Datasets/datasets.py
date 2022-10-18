import numpy as np
import torch
from torch.utils.data import Dataset


class MLP_Dataset(Dataset):
    def __init__(self, data, steps, n_state, n_control, lag):
        self.data = data
        self.steps = steps
        self.n_state = n_state
        self.n_control = n_control
        self.lag = lag
        self.transform = torch.tensor
        self.target_transform = torch.tensor

    def __len__(self):
        return self.data.shape[0] - self.n_state - self.lag - self.steps + 1

    def __getitem__(self, idx):

        x_ = self.data[idx:idx+1, :5]
        u_ = self.data[idx+self.n_state-self.n_control+self.lag: idx+self.n_state+self.lag, 3:5]
        x = np.concatenate((x_, u_), axis=None)    # vx1,vy1,r1, vx2,vy2,r2, ..., u1,d1, u2,d2 ...

        y = self.data[idx+self.n_state: idx+self.n_state+1, -3:]

        if self.transform:
            # cut_x = self.transform(x).cuda()
            x = self.transform(x)

        if self.target_transform:
            # cut_y = self.target_transform(y).cuda()
            y = self.transform(y)

        return x, y


class RNN_Dataset(Dataset):
    def __init__(self, data, steps, n_state, lag):
        self.data = data
        self.steps = steps
        self.n_state = n_state
        self.lag = lag
        self.transform = torch.tensor
        self.target_transform = torch.tensor

    def __len__(self):
        return self.data.shape[0] - self.n_state - self.lag - self.steps + 1

    def __getitem__(self, idx):

        x_ = self.data[idx: idx+1, 0:3*self.n_state].reshape(-1, 3)
        u_ = self.data[idx+self.lag: idx+1+self.lag, 3*self.n_state:5*self.n_state].reshape(-1, 2)
        x = np.concatenate((x_, u_), axis=1)
        y = self.data[idx:idx+1, -3:]

        x = self.transform(x)
        y = self.transform(y)

        return x, y


class zonda_Dataset(Dataset):
    def __init__(self, data, steps, n_state, n_control, lag):
        self.data = data
        self.steps = steps
        self.n_state = n_state
        self.n_control = n_control
        self.transform = torch.tensor
        self.target_transform = torch.tensor
        self.lag = lag

    def __len__(self):
        return self.data.shape[0] - self.n_state - self.lag - self.steps + 1

    def __getitem__(self, idx):

        x_ = self.data[idx: idx+self.n_state, :3]
        u_ = self.data[idx+self.n_state-self.n_control+self.lag: idx+self.n_state+self.lag, 3:5]
        x = np.concatenate((x_, u_), axis=None)

        y = self.data[idx+self.n_state: idx+self.n_state+1, -3:]

        if self.transform:
            # cut_x = self.transform(x).cuda()
            x = self.transform(x)

        if self.target_transform:
            # cut_y = self.target_transform(y).cuda()
            y = self.transform(y)

        return x, y
