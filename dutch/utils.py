import torch
import torch.distributions as dist
torch.manual_seed(0)



def generate_data(feature_dim=5, num_samples=10000):

    z = dist.Bernoulli(0.5).sample([num_samples])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

    t0_t1 = torch.tensor([[0.], [1.]])
    y_t0, y_t1 = dist.Bernoulli(logits=3 * (z + 2 * (2 * t0_t1 - 2))).mean
    true_ite = (y_t1 - y_t0).mean()

    return torch.cat([x, t.unsqueeze(1), y.unsqueeze(1)], dim=1), true_ite


def causal_effect(generator, u, x_arr, test_num):
    it_y = []
    for x in x_arr:
        it_x = x * torch.ones(test_num, 1)
        tmp_y = generator(u, it_x)
        it_y.append(tmp_y[:, -1].mean().item())
    effect = round(it_y[1] - it_y[0], 3)
    return effect
