import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size):
        super().__init__()
        
        self.G_age = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.G_sex = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.G_hidden = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 13),
        )
         
        self.theta_edu = nn.Parameter(0.02 * torch.randn(1))
        self.theta_occupation = nn.Parameter(0.02 * torch.randn(1))
        self.theta_hours = nn.Parameter(0.02 * torch.randn(1))
        self.theta_income = nn.Parameter(0.02 * torch.randn(1))

    def forward(self, u1, u2, u3, is_edu=None):
        age = self.G_age(u1)
        sex = self.G_sex(u2)
        hidden = self.G_hidden(u3)
        
        edu = (hidden[:, :2] * torch.cat((age, sex), dim=1)).sum(dim=1, keepdim=True) + self.theta_edu
        if is_edu is not None:
            edu = is_edu
    
        hours = (hidden[:, 2:5] * torch.cat((age, sex, edu), dim=1)).sum(dim=1, keepdim=True) + self.theta_hours
        occup = torch.sigmoid((hidden[:, 5:8] * torch.cat((age, sex, edu), dim=1)).sum(dim=1, keepdim=True) + self.theta_occupation)
        income = torch.sigmoid((hidden[:, 8:13] * torch.cat((age, sex, edu, hours, occup), dim=1)).sum(dim=1, keepdim=True) + self.theta_income)
        return torch.cat((age, sex, edu, hours, occup, income), dim=1)


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