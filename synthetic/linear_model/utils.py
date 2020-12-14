import torch
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


def gen_linear_data(data_size, is_x=None):
    uz = 0.1 * np.random.randn(data_size, 1)
    ux = 0.1 * np.random.randn(data_size, 1)
    uv = 0.1 * np.random.randn(data_size, 1)
    uw = 0.1 * np.random.randn(data_size, 1)
    uy = 0.1 * np.random.randn(data_size, 1)

    u1 = 0.1 * np.random.randn(data_size, 1)
    u2 = 0.1 * np.random.randn(data_size, 1)
    z = np.random.uniform(1, 5, data_size).reshape(data_size, 1) + uz
    x = 0.9 * z + 3* u1 + ux if is_x is None else is_x
    w = 0.9 * x + 10 * u2 + uw
    v = 0.8 * u1 + 12 * u2 + uv
    y = 1.8 * w + 0.7 *v + uy
    return np.concatenate((x, y, z, w, v), axis=1)


def ground_truth(x_arr, data_size, is_linear):
    it_y = []
    for x in x_arr:
        it_x = x * np.ones((data_size, 1))
        if is_linear:
            data = gen_linear_data(data_size, it_x)
        else:
            data = gen_nonlinear_data(data_size, it_x)
        it_y.append(np.mean(data[:, 1]))
    return it_y


def causal_effect(generator, u1, u2, u3, u4, x_arr, test_num):
    it_y = []
    for x in x_arr:
        it_x = x * torch.ones(test_num, 1)
        tmp_y = generator(u1, u2, u3, u4, it_x)
        it_y.append(tmp_y[:, 1].mean().item())
    effect = [round(y1 - it_y[0], 3) for y1 in it_y[1:]]
    return effect