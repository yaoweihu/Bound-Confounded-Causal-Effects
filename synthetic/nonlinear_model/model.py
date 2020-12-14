import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size, x_size, w_size, y_size):
        super().__init__()

        self.x_size = x_size
        self.w_size = w_size
        self.y_size = y_size

        self.Gz = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.Gx = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * x_size)
        )
        self.Gx_bias = nn.Parameter(torch.zeros(1))
        self.Gw = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * w_size)
        )
        self.Gw_bias = nn.Parameter(torch.zeros(1))
        self.Gv = nn.Sequential(
            nn.Linear(2*noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.Gy = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4 * y_size)
        )
        self.Gy_bias = nn.Parameter(torch.zeros(1))

    def forward(self, u1, u2, u3, u4, is_x=None):
        Z = self.Gz(u1)
        Rx = self.Gx(u2)
        Rw = self.Gw(u3)
        V = self.Gv(torch.cat((u2, u3), dim=1))
        Ry = self.Gy(u4)
        
        X = F.relu(Rx[:, :self.x_size].mul(Z))
        X = X.mul(Rx[:, self.x_size:]).add(self.Gx_bias)
        X = X.sum(dim=1, keepdim=True)
        if is_x is not None:
            X = is_x
        
        W = F.relu(Rw[:, :self.w_size].mul(X))
        W = W.mul(Rw[:, self.w_size:]).add(self.Gw_bias)
        W = W.sum(dim=1, keepdim=True)
        
        Y1 = F.relu(Ry[:, : self.y_size].mul(W))
        Y1 = Y1.mul(Ry[:, self.y_size : 2*self.y_size])
        Y2 = F.relu(Ry[:, 2*self.y_size: 3*self.y_size].mul(V))
        Y2 = Y2.mul(Ry[:, 3*self.y_size :])
        Y =  Y1.add(Y2).add(self.Gy_bias)
        Y = Y.sum(dim=1, keepdim=True)

        return torch.cat((X, Y, Z, W, V), dim=1)


class WGAN_Discriminator(nn.Module):
    def __init__(self, num_nodes, hidden_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_nodes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data):
        return self.model(data)