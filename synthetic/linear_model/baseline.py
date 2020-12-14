import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import utils
np.random.seed(0)



def normal(x, mu, var):
    return 1/np.sqrt(2 * np.pi * var) * np.exp(-1 / (2*var) * (x-mu)**2)


if __name__ == "__main__":
    
    data_size = data_size = 10000
    test_num = 10000
    int_x = [0, 1]

    print("x0={}, x1={}".format(int_x[0], int_x[1]))
    ######################## data ##################################
    data = utils.gen_linear_data(data_size)

    ######################## simulated ground truth ################
    int_y = utils.ground_truth(int_x, test_num, is_linear=True)
    effect = [round(y1 - int_y[0], 3) for y1 in int_y[1:]]
    print("simulated: {:.3f}".format(effect[0]))

    ####################### linear regression ######################
    X = data[:, 0][:, np.newaxis]
    W1 = data[:, 3]
    reg1 = LinearRegression().fit(X, W1)
    W2 = data[:, 3:5]
    Y = data[:, 1][:, np.newaxis]
    reg2 = LinearRegression().fit(W2, Y)
    print("linear regression: {:.3f}".format((int_x[1] - int_x[0]) * reg1.coef_[0] * reg2.coef_[0][0]))

    ####################### instrumental variable ##################
    X, Y, Z = data[:, 0][:], data[:, 1], data[:, 2]
    array_YZ = np.vstack((Y, Z))
    cov_YZ = np.cov(array_YZ)[0][1]
    array_XZ = np.vstack((X, Z))
    cov_XZ = np.cov(array_XZ)[0][1]
    print("instrumental variable: {:.3f}".format((int_x[1] - int_x[0]) * cov_YZ / cov_XZ))

    ###################### propensity score ########################
    X, Y, Z = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
    W1, W2 = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis]
    #### step 1 ####
    fan = np.hstack((np.ones_like(X), Z, W1, W2))
    reg1 = LinearRegression(fit_intercept=False).fit(fan, X)
    Beta = reg1.coef_.transpose()
    variance = np.mean((X - np.dot(fan, Beta))**2)
    R = normal(X, X - np.dot(fan, Beta), variance)
    #### step 2 ####
    inpt = np.hstack((np.ones_like(X), X,  R, X*R))
    reg2 = LinearRegression(fit_intercept=False).fit(inpt, Y)
    Alpha = reg2.coef_.transpose()
    #### step3 ####
    t0 = int_x[0] * np.ones_like(X)
    r0 = normal(t0, np.dot(fan, Beta), variance)
    inpt0 = np.hstack((np.ones_like(X), t0, r0, t0*r0))
    y0 = np.mean(np.dot(inpt0, Alpha))

    t1 = int_x[1] * np.ones_like(X)
    r1 = normal(t1, np.dot(fan, Beta), variance)
    inpt1 = np.hstack((np.ones_like(X), t1, r1, t1*r1))
    y1 = np.mean(np.dot(inpt1, Alpha))
    print("Propensity score: {:.3f}".format(y1-y0))