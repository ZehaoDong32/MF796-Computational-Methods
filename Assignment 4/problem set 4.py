import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.style.use('seaborn')

# Problem 1
tickers = pd.read_csv("stocklist.csv").tickers.values.tolist()
data = yf.download(tickers, start='2015-01-01', end='2020-01-01')
data = data['Adj Close']
# check nulls
print('Is there any NULL?\n', data.isnull().any().any())
# use last valid observation to fill gap
data = data.fillna(method='ffill')
print('Is there any NULL?\n', data.isnull().any().any())
ret_data = np.log(data.pct_change() + 1).dropna()
cov = ret_data.cov().values
e_vals, e_vecs = np.linalg.eig(cov)
print('number of positive eigenvalues:', sum(e_vals > 0))
print('number of negative eigenvalues:', sum(e_vals < 0))
print('number of zero eigenvalues:', sum(e_vals == 0))
e_vals.sort()
e_vals = e_vals[::-1]
plt.scatter(np.arange(len(e_vals)), e_vals)
plt.ylim([1.25 * np.min(e_vals) - 0.25 * np.max(e_vals), -0.25 * np.min(e_vals) + 1.25 * np.max(e_vals)])
plt.title('Eigenvalues in Order', fontsize=18)
plt.ylabel('Eigenvalue', fontsize=18)
plt.show()
i = 0
while i < 100:
    if e_vals[0:i].sum() / e_vals.sum() > 0.5:
        break
    i += 1
print('{} eigenvalues are required to account for 50% of the variance'.format(i + 1))
i = 0
while i < 100:
    if e_vals[0:i].sum() / e_vals.sum() > 0.9:
        break
    i += 1
print('{} eigenvalues are required to account for 90% of the variance'.format(i + 1))
pca = PCA(n_components=i + 1)
factor_ret = pca.fit_transform(ret_data)
ret_mat = ret_data.values
beta = np.dot(np.dot(np.linalg.inv(np.dot(factor_ret.T, factor_ret)), factor_ret.T), ret_mat)
residual_ret = ret_mat - np.dot(factor_ret, beta)
residual_data = pd.DataFrame(residual_ret, index=ret_data.index)
plt.title('Residual Return', fontsize=18)
plt.ylabel('Returns', fontsize=18)
plt.plot(residual_data)
plt.show()
print('The mean variance of raw return data is:', ret_data.var().mean())
print('The mean variance of factor return is:', factor_ret.var().mean())
print('The mean variance of residual return is:', residual_data.var().mean())

# Problem 2
G = np.matrix(np.ones((2, 100)))
G[1, 17:100] = 0
U, d, V = np.linalg.svd(cov)
U = np.matrix(U)
V = np.matrix(V)
D = np.matrix(np.diag(d))
C_inv = U * D.I * V
a = 5
c = np.matrix([1, 0.1]).T
R = np.matrix(ret_data.mean().values).T
Lambda = (G * C_inv * G.T).I * (G * C_inv * R - 2 * a * c)
w = 1 / 2 / a * C_inv * (R - G.T * Lambda)
plt.title('Weights of Each Stock', fontsize=18)
plt.ylabel('Weights', fontsize=18)
plt.plot(w)
plt.show()
print('number of positive weights are:', int(sum(w >= 0)))
print('number of negative weights are:', int(sum(w < 0)))
