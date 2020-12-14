import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli

class Generator(nn.Module):
    def __init__(self, noise_size, hidden_size):
        super().__init__()

        self.G_age = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.G_birth = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.G_marital = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )
        self.G_edu = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )
        self.G_occupation = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, u1, u2, u3, is_t=None):
        age = self.G_age(u1)
        birth = self.G_birth(u2)

        h_marital = self.G_marital(u3)
        age_birth = torch.cat([age, birth, torch.ones(age.shape)], dim=1)
        marital = (h_marital * age_birth).sum(dim=1, keepdim=True)

        h_edu = self.G_edu(u3)
        age_birth_marital = torch.cat([marital, age_birth], dim=1)
        edu = (h_edu * age_birth_marital).sum(dim=1, keepdim=True)
        if is_t is not None:
            edu = is_t

        h_occupation = self.G_occupation(u3)
        age_edu = torch.cat([age, edu, torch.ones(age.shape)], dim=1)
        occupation = torch.sigmoid((h_occupation * age_edu).sum(dim=1, keepdim=True))
        
        return torch.cat((age, birth, marital, edu, occupation), dim=1)


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