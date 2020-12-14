import torch
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import Generator, WGAN_Discriminator

np.random.seed(0)
torch.manual_seed(0)

# ------------------------------------------ data ------------------------------------------------
data = pd.read_csv("../data/adult_f.csv", header=0, index_col=False)
data = np.array(data[['Age', 'Sex', 'Education', 'Hours_Per_Week', 'Occupation', 'Income']])

# ------------------------------------------- parameters ------------------------------------------
noise_size = 16
hidden_size = 16
num_nodes = 6
test_num = 2000

lr = 3e-4
lmd_lr = 5e-5
clip_value = 0.02
epochs = 2000
batch_size = 512
n_crite = 1
tao = 0.00002

max_flag = False
int_x = [10/16, 16/16]
# ------------------------------------------ model ------------------------------------------------
generator = Generator(noise_size, hidden_size)
discriminator = WGAN_Discriminator(num_nodes, hidden_size)

lmd = torch.tensor(0.1, requires_grad=True)
l_optimizer = optim.RMSprop([lmd], lr=lmd_lr)
g_optimizer = optim.RMSprop(generator.parameters(), lr=lr)
d_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)

# ------------------------------------------training-----------------------------------------------
effect_int = []
dd_list, gg_list, ll_list = [], [], []
for epoch in range(1, epochs+1, 1):
    np.random.shuffle(data)
    iter_num = len(data) // batch_size

    d_list, g_list= [], []
    for i in range(iter_num):
        batch = torch.FloatTensor(data[batch_size * i: batch_size * (i + 1)])

        U1 = torch.randn(batch_size, noise_size)
        U2 = torch.randn(batch_size, noise_size)
        U3 = torch.randn(batch_size, noise_size)

        fake_data = generator(U1, U2, U3)

        ############# Discriminator #############
        real_score = torch.mean(discriminator(batch))
        fake_score = -torch.mean(discriminator(fake_data.detach()))
        d_loss = -(real_score + fake_score)
        d_list.append(-d_loss.item())
        
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        for p in discriminator.parameters():
           p.data.clamp_(-clip_value, clip_value)

        ############## Generator #################
        if i % n_crite == 0:

            tmp_effect = []
            x0 = int_x[0] * torch.ones(batch_size, 1)
            x1 = int_x[1] * torch.ones(batch_size, 1)
            y0 = generator(U1, U2, U3, x0)
            y1 = generator(U1, U2, U3, x1)
            if max_flag:
                mn_loss = torch.mean(y0[:, 5] - y1[:, 5])
            else:
                mn_loss = torch.mean(y1[:, 5] - y0[:, 5])

            g_loss = -torch.mean(discriminator(fake_data))
            m_g_loss = lmd.detach() * g_loss
            loss = mn_loss + lmd * (tao + d_loss.detach())
            g_list.append(loss.item())    

            g_optimizer.zero_grad()
            m_g_loss.backward()
            for param in generator.parameters():
                g = torch.autograd.grad(mn_loss, param, retain_graph=True, allow_unused=True)[0]
                if g is not None:
                    p_norm = torch.norm(param.grad, 2)
                    g_norm = torch.norm(g.data, 2)
                    ga = 0.04 * torch.min(p_norm, g_norm) / (g_norm + 1e-12) * g.data
                    param.grad.add_(ga)
            g_optimizer.step()

            l_optimizer.zero_grad()
            loss.backward()
            l_optimizer.step()

    if  lmd.detach().item() <= 0.001 and lmd.detach().item() >= -0.001:
        U1 = torch.randn(test_num, noise_size)
        U2 = torch.randn(test_num, noise_size)
        U3 = torch.randn(test_num, noise_size)

        fake_data_0 = generator(U1, U2, U3, int_x[0] * torch.ones(test_num, 1))
        fake_data_1 = generator(U1, U2, U3, int_x[1] * torch.ones(test_num, 1))
        incomes_0 = fake_data_0[:, 5].mean().item()
        incomes_1 = fake_data_1[:, 5].mean().item()
        fake_effect = incomes_1 - incomes_0
        effect_int.append(fake_effect)
    if len(effect_int) >= 50:
        break

    dd_list.append(np.mean(d_list))
    gg_list.append(np.mean(g_list))
    ll_list.append(lmd.detach().item())
    print("Epoch: {}, D Loss: {:.10f}, G Loss: {:.10f}".format(epoch, np.mean(d_list), np.mean(g_list)))

print("mean: {:.10f}, var: {:.10f}".format(np.mean(effect_int), np.var(effect_int)))