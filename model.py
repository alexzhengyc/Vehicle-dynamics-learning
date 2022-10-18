import torch
from torch import nn


class MLP2(nn.Module):

    def __init__(self, neurons, steps, n_state, n_control):
        super(MLP2, self).__init__()
        self.device = 'cuda'
        self.neurons = neurons
        self.steps = steps
        self.n_state = n_state
        self.n_control = n_control

        self.f = nn.Sequential(
            nn.Linear(3*self.n_state+2*self.n_control, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, 3),
            nn.Sigmoid()
        )

    def forward(self, x):

        y_pred = self.f(x.unsqueeze(1))

        return y_pred


class LSTM(nn.Module):

    def __init__(self, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.device = 'cuda'
        self.in_dim = 5
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.in_dim, self.hidden_dim, self.num_layers, batch_first=True)

        self.f = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x):

        output, (hn, cn) = self.lstm(x)
        # print(output.shape)
        y_pred = self.f(output)[:, -1, :].unsqueeze(1)
        # print(y_pred.shape)

        return y_pred


class MLP2plus(nn.Module):

    def __init__(self, neurons, steps, n_state, n_control):
        super(MLP2plus, self).__init__()
        self.device = 'cuda'
        self.steps = steps
        self.n_state = n_state
        self.n_control = n_control
        self.neurons = neurons

        self.fx = nn.Sequential(
            nn.Linear(self.n_state, 2),
            nn.ReLU()
        )
        self.fu = nn.Sequential(
            nn.Linear(self.n_control, 2),
            nn.ReLU()
        )
        self.f = nn.Sequential(
            nn.Linear(10, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, 3),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [ x, y, psi, t, s] * self.steps + 2

        batch = x.shape[0]
        l1 = torch.zeros(batch, 10)
        for i in range(batch):
            vx_ = x[i][0:3*self.n_state:3]
            vy_ = x[i][1:3*self.n_state:3]
            r_ = x[i][2:3 * self.n_state:3]
            throttle_ = x[i][3*self.n_state::2]
            steering_ = x[i][3*self.n_state+1::2]
            vx = self.fx(vx_)
            vy = self.fx(vy_)
            r = self.fx(r_)
            throttle = self.fu(throttle_)
            steering = self.fu(steering_)
            l1[i] = torch.concat([vx, vy, r, throttle, steering])

        y_pred = self.f(l1.unsqueeze(1))
        return y_pred

