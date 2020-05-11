import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def BS_model(K, S, sigma, r, T, q=0):
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d2 = d1 - sigma * T ** 0.5
    return (norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T))


def discretization(T, S0, r, smax, N, M, K1, sigma1, K2=None, sigma2=None, option='European'):
    ht = T / N
    hs = smax / M
    if K2 == None:
        sigma = sigma1
    else:
        sigma = (sigma1 + sigma2) / 2

    # Matrix A
    s = np.arange(0, smax + hs, hs)
    a = 1 - sigma ** 2 * s ** 2 * ht / hs ** 2 - r * ht
    l = sigma ** 2 * s ** 2 * ht / hs ** 2 / 2 - r * s * ht / hs / 2
    u = sigma ** 2 * s ** 2 * ht / hs ** 2 / 2 + r * s * ht / hs / 2
    A = np.diag(a[1:M])
    l_list = l[2:M]
    u_list = u[1:M - 1]
    for i in range(len(l_list)):
        A[i + 1][i] = l_list[i]
        A[i][i + 1] = u_list[i]
    eig_vals, eig_vecs = np.linalg.eig(A)

    if K2 == None:
        diff = s - K1
        diff[diff < 0] = 0
        c = diff[1:M]
        b = np.zeros(M - 1)
        for i in range(N):
            b[-1] = u[-2] * (smax - K1 * np.exp(-r * i * ht))
            if option == 'American':
                c = [max(x, y) for x, y in zip(c, diff[1:M])]
            c = A.dot(c) + b
        c0_BS = BS_model(K1, S0, sigma, r, T, q=0)
    else:
        long = s - K1
        long[long < 0] = 0
        short = s - K2
        short[short < 0] = 0
        diff = long - short
        c = diff[1:M]
        b = np.zeros(M - 1)
        for i in range(N):
            b[-1] = u[-2] * (K2 - K1) * np.exp(-r * i * ht)
            if option == 'American':
                c = [max(x, y) for x, y in zip(c, diff[1:M])]
            c = A.dot(c) + b
        c0_BS = BS_model(K1, S0, sigma, r, T, q=0) - BS_model(K2, S0, sigma, r, T, q=0)

    c0 = np.interp(S0, s[1:M], c)
    error = abs((c0 - c0_BS) / c0_BS)

    return c0, error, eig_vals, eig_vecs


SPY = 312.86

T = 145 / 252
S = 312.86
K1 = 315
K2 = 320
r = 0.72 / 100
smax = 630
sigma1 = 22.1 / 100
sigma2 = 21.22 / 100
sigma = (sigma1 + sigma2) / 2
N = 2665
M = 315

# print('Minimum Nt:', T / (1 / (sigma ** 2 * smax ** 2) * hs ** 2))  # min Nt

eig__, _, eig_vals, eig_vecs = discretization(T, S, r, smax, N, M, K1, sigma1, K2, sigma2, option='European')

# plt.figure(figsize=(10, 6))
# plt.plot(eig_vals)
# plt.title('Eigenvalues')
# plt.ylabel('Eigenvalues')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.plot(abs(eig_vals))
# plt.title('Absolute Eigenvalues')
# plt.ylabel('Absolute Eigenvalues')
# plt.show()

print('Is all eigenvalues between -1 and 1 ?\n', any(abs(eig_vals <= 1)))

callspread_euro = discretization(T, S, r, smax, N, M, K1, sigma1, K2, sigma2, option='European')[0]
print('price of the call spread without the right of early exercise is:', callspread_euro)
# error = discretization(T,S,r,smax,N,M,K1,sigma1,K2, sigma2,option='European')[1]
# print(error)

callspread_amer = discretization(T, S, r, smax, N, M, K1, sigma1, K2, sigma2, option='American')[0]
print('price of the call spread with the right of early exercise is:', callspread_amer)

premium = callspread_amer - callspread_euro
print('The early exercise premium as the difference between the American and European call spreads is:',
      premium, '\nabout {} of the European call spread'.format(premium/callspread_euro))
