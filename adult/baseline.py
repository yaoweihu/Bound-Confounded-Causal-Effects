import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(0)


# --------------------- load data -----------------------------------------------------------
ori_data = pd.read_csv("../data/adult_f.csv", header=0, index_col=False)


X0 = 1
X1 = 16

# --------------------- linear regression ----------------------------------------------------
data = np.array(ori_data[['Age', 'Sex', 'Education', 'Hours_Per_Week', 'Occupation', 'Income']])
variables = np.array(ori_data[['Age', 'Sex', 'Education', 'Hours_Per_Week', 'Occupation']])
incomes = np.array(ori_data['Income'])

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(variables, incomes)

x0 = variables
x0[:, 2] = X0 / 16
scores0 = clf.predict_proba(x0)
y0 = np.mean(scores0[:, 1])

x1 = variables
x1[:, 2] = X1 / 16
scores1 = clf.predict_proba(x1)
y1 = np.mean(scores1[:, 1])

edu_effect = y1 - y0
print("edu - linear regression: {:.5f}".format(edu_effect))


# # --------------------- propensity score ------------------------------------------------------

covariances = np.array(ori_data[['Age', 'Sex', 'Hours_Per_Week', 'Occupation']])
education = np.array(ori_data['Education'] * 16, np.int32)
incomes = np.array(ori_data['Income'])

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000).fit(covariances, education)
scores = clf.predict_proba(covariances)

index0 = np.where(education == X0)
y0 = np.sum(incomes[index0] /scores[index0][:, X0-1]) / np.sum(1/scores[index0][:, X0-1])

index1 = np.where(education == X1)
y1 = np.sum(incomes[index1] /scores[index1][:, X1-1]) / np.sum(1/scores[index1][:, X1-1])

effect = y1 - y0
print("edu - propensity score: {:.5f}".format(effect))