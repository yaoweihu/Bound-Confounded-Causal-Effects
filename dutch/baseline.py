import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
np.random.seed(0)

X0 = 4
X1 = 5

##------------------------------- ground truth ----------------------------------------------
data = pd.read_csv("../data/Dutch.csv", header=0, index_col=False)
data['occupation'] = data['occupation'].map({'5_4_9': 0, '2_1': 1})
data = data[['sex', 'age', 'countrybirth', 'maritial', 'edu', 'occupation']]
#print(data[data['edu'] == 5]['occupation'].value_counts())
#print(data['edu'].value_counts())
# 1 and 2
p_s = data['sex'].value_counts(normalize=True)
# 0 and 1
p_a = data['age'].value_counts(normalize=True)
# 1 - 3
p_c = data['countrybirth'].value_counts(normalize=True)
# 1 - 4
def p_m_cas(country, age, sex):
    res = data[(data['countrybirth'] == country) & (data['age'] == age) & (data['sex'] == sex)]
    return res['maritial'].value_counts(normalize=True)
# 0 and 1
def p_o_eas(edu, age, sex):
    res = data[(data['edu'] == edu) & (data['age'] == age) & (data['sex'] == sex)]
    return res['occupation'].value_counts(normalize=True)[1]

def p_do_e(edu):
    effect = 0
    for sex in [1, 2]:
        for age in [0, 1]:
            for country in [1, 2, 3]:
                for maritial in [1, 2, 3, 4]:
                    effect += (p_s[sex] * p_a[age] * p_c[country] * p_m_cas(country, age, sex)[maritial] * p_o_eas(edu, age, sex))
    return effect

effect = p_do_e(X1) - p_do_e(X0)
print("ground truth:", effect)


# --------------------- linear regression ----------------------------------------------------
variables = np.array(data[['age', 'countrybirth', 'maritial', 'edu']])
occupation = np.array(data['occupation'])

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(variables, occupation)

x0 = variables
x0[:, -1] = X0
scores0 = clf.predict_proba(x0)
y0 = np.mean(scores0[:, 1])

x1 = variables
x1[:, -1] = X1
scores1 = clf.predict_proba(x1)
y1 = np.mean(scores1[:, 1])

edu_effect = y1 - y0
print("edu - linear regression: {:.5f}".format(edu_effect))



# # --------------------- propensity score ------------------------------------------------------
covariances = np.array(data[['age', 'countrybirth', 'maritial']])
edu = np.array(data['edu'])
occupation = np.array(data['occupation'])
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=10000).fit(covariances, edu)
scores = clf.predict_proba(covariances)

index0 = np.where(edu == X0)
#y0 = np.mean(occupation[index0] / (1 - scores[index0][:, X0]))
y0 = np.sum(occupation[index0] /scores[index0][:, X0-1]) / np.sum(1/scores[index0][:, X0-1])

index1 = np.where(edu == X1)
#y1 = np.mean(occupation[index1] / scores[index1][:, X1])
y1 = np.sum(occupation[index1] /scores[index1][:, X1-1]) / np.sum(1/scores[index1][:, X1-1])


effect = y1 - y0
print("edu - propensity score: {:.5f}".format(effect))


# ========================= instrumental variable ==============================
X, Y, Z = data[['edu']], data[['occupation']], data[['countrybirth']]
array_YZ = np.hstack((Y, Z)).transpose()
cov_YZ = np.cov(array_YZ)[0][1]
array_XZ = np.hstack((X, Z)).transpose()
cov_XZ = np.cov(array_XZ)[0][1]
print("instrumental variable: {:.3f}".format((X1 - X0) * cov_YZ / cov_XZ))
