import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size):
        super().__init__()
        
        self.Gz = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.Gx = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.theta_x = nn.Parameter(0.02 * torch.randn(1))
        self.Gw = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.theta_w = nn.Parameter(0.02 * torch.randn(1))
        self.Gv = nn.Sequential(
            nn.Linear(2*noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.Gy = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.theta_y = nn.Parameter(0.02 * torch.randn(2))

    def forward(self, u1, u2, u3, u4, is_x=None):
        Z = self.Gz(u1)
        Rx = self.Gx(u2)
        Rw = self.Gw(u3)
        V = self.Gv(torch.cat((u2, u3), dim=1))
        Ry = self.Gy(u4)
        
        X = self.theta_x * Z + Rx
        if is_x is not None:
            X = is_x
        
        W = self.theta_w * X + Rw
        Y = self.theta_y[0] * W + self.theta_y[1] * V + Ry
        
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