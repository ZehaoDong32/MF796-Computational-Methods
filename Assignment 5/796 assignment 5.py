from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize
import cmath
import warnings
from scipy.optimize import root

plt.style.use('seaborn')


# problem 1
def K_call(sigma, T, delta, S0=100, r=0):
    K = S0 / np.exp(norm.ppf(delta) * sigma * np.sqrt(T) - (r + sigma ** 2 / 2) * T)
    return K


def K_put(sigma, T, delta, S0=100, r=0):
    K = S0 / np.exp(norm.ppf(1 - delta) * sigma * np.sqrt(T) - (r + sigma ** 2 / 2) * T)
    return K


def BS_model(K, sigma, T, S0=100, r=0, q=0):
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    d2 = d1 - sigma * T ** 0.5
    return norm.cdf(d1) * S0 * np.exp(-q * T) - norm.cdf(d2) * K * np.exp(-r * T)


def density(T, K=100, r=0):
    density = []
    sigma_list = []
    K_list = np.linspace(K - 12, K + 12, int(24.1 / 0.1))
    if T == 1 / 12:
        sigma_list = const1 + coef1 * K_list
    else:
        sigma_list = const2 + coef2 * K_list
    for i in range(1, len(K_list) - 1):
        c1 = BS_model(K_list[i - 1], sigma_list[i - 1], T)
        c2 = BS_model(K_list[i], sigma_list[i], T)
        c3 = BS_model(K_list[i + 1], sigma_list[i + 1], T)
        density += [(c1 - 2 * c2 + c3) / 0.1 ** 2]

    return K_list, density


def density2(T, K=100, r=0):
    density = []
    K_list = np.linspace(K - 30, K + 30, int(60.1 / 0.1))
    if T == 1 / 12:
        sigma = 0.1824
    else:
        sigma = 0.1645
    for i in range(1, len(K_list) - 1):
        c1 = BS_model(K_list[i - 1], sigma, T)
        c2 = BS_model(K_list[i], sigma, T)
        c3 = BS_model(K_list[i + 1], sigma, T)
        density += [(c1 - 2 * c2 + c3) / 0.1 ** 2]

    return K_list, density


# Problem 2
def check_monotone(df, opt_type='call'):
    res = True
    if type is 'call':
        res = any(df.groupby('expDays')['mid_price'].pct_change().dropna() <= 0)
    elif type is 'put':
        res = any(df.groupby('expDays')['mid_price'].pct_change().dropna() >= 0)
    return res


def check_rate(df, opt_type='call'):
    res = []
    for T in df.expDays.unique():
        temp = df[df.expDays == T]
        r = ((temp['mid_price'].shift(1) - temp['mid_price']) / (temp['K'].shift(1) - temp['K'])).dropna().values
        if opt_type is 'call':
            res.append(True if any(r > -1) and any(r < 0) else False)
        elif opt_type is 'put':
            res.append(True if all(r > 0) and all(r < 1) else False)
    return any(res)


def check_convexity(df):
    res = []
    for T in df_call.expDays.unique():
        temp = df[df.expDays == T]
        res.append(any((temp['mid_price'] - 2 * temp['mid_price'].shift(1) + temp['mid_price'].shift(2)).dropna() > 0))
    return res


class FFT:
    def __init__(self, params, T, S0=267.15, r=0.015, q=0.0177):
        self.kappa = params[0]
        self.theta = params[1]
        self.sigma = params[2]
        self.rho = params[3]
        self.v0 = params[4]

        self.T = T
        self.S0 = S0
        self.r = r
        self.q = q

    def Heston_fft(self, alpha, n, B, K):
        ii = complex(0, 1)
        N = 2 ** n
        dv = B / N
        dk = 2 * np.pi / N / dv
        beta = np.log(self.S0) - dk * N / 2

        vj = np.arange(0, N, dtype=complex) * dv
        km = beta + np.arange(0, N) * dk  # ln(K)

        delta_j_1 = np.zeros(N)
        delta_j_1[0] = 1

        Psi_vj = np.zeros(N, dtype=complex)

        for j in range(0, N):
            u = vj[j] - (alpha + 1) * ii
            numer = np.exp(-ii * beta * vj[j]) * self.Heston_cf(u)
            denom = 2 * (alpha + vj[j] * ii) * (alpha + 1 + vj[j] * ii)
            Psi_vj[j] = numer / denom

        x = (2 - delta_j_1) * dv * Psi_vj
        z = np.fft.fft(x)

        Mul = np.exp(-alpha * np.array(km)) / np.pi
        Calls = np.exp(-self.r * self.T) * Mul * np.array(z).real

        K_list = list(np.exp(km))
        Call_list = list(Calls)
        tck = interpolate.splrep(K_list, Call_list)
        price = interpolate.splev(K, tck).real
        return price

    def Heston_cf(self, u):
        sigma = self.sigma
        kappa = self.kappa
        rho = self.rho
        S0 = self.S0
        r = self.r
        T = self.T
        theta = self.theta
        v0 = self.v0

        ii = complex(0, 1)

        lmbd = cmath.sqrt(sigma ** 2 * (u ** 2 + ii * u) +
                          (kappa - ii * rho * sigma * u) ** 2)
        w_nume = np.exp(ii * u * np.log(S0) + ii * u * (r - self.q) * T +
                        kappa * theta * T * (kappa - ii * rho * sigma * u) / sigma ** 2)
        w_deno = (cmath.cosh(lmbd * T / 2) + (kappa - ii * rho * sigma * u) /
                  lmbd * cmath.sinh(lmbd * T / 2)) ** (2 * kappa * theta / sigma ** 2)
        w = w_nume / w_deno
        y = w * np.exp(-(u ** 2 + ii * u) * v0 / (lmbd /
                                                  cmath.tanh(lmbd * T / 2) + kappa - ii * rho * sigma * u))

        return y


def cal_price(params, alpha, n, B, K, T):
    prices = FFT(params, T).Heston_fft(alpha, n, B, K)
    return prices


def sqr_sum(df, params, alpha, n, B):
    res = 0
    for T in df.expT.unique():
        temp = df[df.expT == T]
        k = temp.K
        prices = cal_price(params, alpha, n, B, k, T)
        res += sum((prices - temp.mid_price) ** 2)
    return res


def fun(params, df_call, df_put, alpha, n, B):
    res = 0
    res += sqr_sum(df_call, params, alpha, n, B)
    res += sqr_sum(df_put, params, -alpha, n, B)
    return res


def callbackF(Xi):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, fun(Xi, df_call, df_put, alpha, n, B)))
    times += 1


def sqr_sum2(df, params, alpha, n, B, opt_type='call'):
    res = 0
    for T in df.expT.unique():
        temp = df[df.expT == T]
        w = 1 / (temp[opt_type + '_ask'] - temp[opt_type + '_bid'])
        k = temp.K
        prices = cal_price(params, alpha, n, B, k, T)
        res += sum(w * (prices - temp.mid_price) ** 2)
    return res


def fun2(params, df_call, df_put, alpha, n, B):
    res = 0
    res += sqr_sum2(df_call, params, alpha, n, B)
    res += sqr_sum2(df_put, params, -alpha, n, B, 'put')
    return res


def callbackF2(Xi):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, fun2(Xi, df_call, df_put, alpha, n, B)))
    times += 1


# problem 3
def Heston_delta(params, K=275, S0=267.15, h=0.01, T=1 / 4):
    p1 = FFT(params, T, S0 + h).Heston_fft(1.5, 12, 1000, K)
    p2 = FFT(params, T, S0 - h).Heston_fft(1.5, 12, 1000, K)
    delta = (p1 - p2) / 2 / h
    return delta


def BS_delta(sigma, K=275, T=1 / 4, S0=267.15, r=0.015, q=0.0177):
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    delta = np.exp(-q * T) * norm.cdf(d1)
    return delta


def Heston_vega(params, K=275, S0=267.15, h=0.01, T=1 / 4):
    params1 = np.array(params) + np.array([0, 0.01, 0, 0, 0.01])
    params2 = np.array(params) - np.array([0, 0.01, 0, 0, 0.01])
    p1 = FFT(params1, T, S0).Heston_fft(1.5, 12, 1000, K)
    p2 = FFT(params2, T, S0).Heston_fft(1.5, 12, 1000, K)
    vega = (p1 - p2) / 2 / h
    return vega


def BS_vega(sigma, K=275, T=1 / 4, S0=267.15, r=0.015, q=0.0177):
    d1 = (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** 0.5)
    vega = S0 * np.exp(-q * T) * np.sqrt(T) / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 / 2)
    return vega


# problem 1

K_dict = {}
K_dict['10DP'] = [K_put(0.3225, 1 / 12, 0.1), K_put(0.2836, 3 / 12, 0.1)]
K_dict['25DP'] = [K_put(0.2473, 1 / 12, 0.25), K_put(0.2178, 3 / 12, 0.25)]
K_dict['40DP'] = [K_put(0.2021, 1 / 12, 0.4), K_put(0.1818, 3 / 12, 0.4)]
K_dict['50D'] = [K_put(0.1824, 1 / 12, 0.5), K_put(0.1645, 3 / 12, 0.5)]
K_dict['40DC'] = [K_call(0.1574, 1 / 12, 0.4), K_call(0.1462, 3 / 12, 0.4)]
K_dict['25DC'] = [K_call(0.1370, 1 / 12, 0.25), K_call(0.1256, 3 / 12, 0.25)]
K_dict['10DC'] = [K_call(0.1148, 1 / 12, 0.1), K_call(0.1094, 3 / 12, 0.1)]

K_table = pd.DataFrame.from_dict(K_dict, orient='index', columns=['1M', '3M'])
print(K_table)

sigma_1m = [0.3225, 0.2473, 0.2021, 0.1824, 0.1574, 0.1370, 0.1148]

sigma_3m = [0.2836, 0.2178, 0.1818, 0.1645, 0.1462, 0.1256, 0.1094]

x1 = K_table['1M']
y1 = sigma_1m
coef1 = np.polyfit(x1, y1, 1)[0]
const1 = np.polyfit(x1, y1, 1)[1]
print(np.polyfit(x1, y1, 1))

x2 = K_table['3M']
y2 = sigma_3m
coef2 = np.polyfit(x2, y2, 1)[0]
const2 = np.polyfit(x2, y2, 1)[1]
print(np.polyfit(x2, y2, 1))

plt.figure(figsize=(12, 8))
plt.plot(density(1 / 12)[0][1:-1], density(1 / 12)[1])
plt.plot(density(3 / 12)[0][1:-1], density(3 / 12)[1])
plt.legend(['1M', '3M'])
plt.title("Risk Neutral Density for 1M & 3M Options")
plt.xlabel("K")
plt.ylabel("Density")
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(density2(1 / 12)[0][1:-1], density2(1 / 12)[1])
plt.plot(density2(3 / 12)[0][1:-1], density2(3 / 12)[1])
plt.legend(['1M', '3M'])
plt.title("Risk Neutral Density for 1M & 3M Options With Constant Sigma")
plt.xlabel("K")
plt.ylabel("Density")
plt.show()

K_list, fi = density(1 / 12)[0][1:-1], density(1 / 12)[1]
temp = interpolate.splrep(K_list, fi)

K_list2, fi2 = density(3 / 12)[0][1:-1], density(3 / 12)[1]
temp2 = interpolate.splrep(K_list2, fi2)

price1 = sum(temp[1][temp[0] <= 110] * 0.1)
print('Price of 1M European Digital Put Option with Strike 110 is:', price1)

price2 = sum(temp2[1][temp2[0] >= 105] * 0.1)
print('Price of 3M European Digital Call Option with Strike 105 is:', price2)

fi3 = []
K_temp = np.linspace(100 - 12, 100 + 12, int(24.1 / 0.1))
sigma_list3 = (const1 + const2) / 2 + (coef1 + coef2) / 2 * K_temp
for i in range(1, len(K_temp) - 1):
    c1 = BS_model(K_temp[i - 1], sigma_list3[i - 1], 2 / 12)
    c2 = BS_model(K_temp[i], sigma_list3[i], 2 / 12)
    c3 = BS_model(K_temp[i + 1], sigma_list3[i + 1], 2 / 12)
    fi3 += [(c1 - 2 * c2 + c3) / 0.1 ** 2]

temp3 = interpolate.splrep(K_list, fi3)

price3 = sum(temp3[1][temp3[0] - 100 > 0] * (temp3[0][temp3[0] - 100 > 0] - 100) * 0.1)
print('Price of 2M European Call Option with Strike 100 is:', price3)

# problem 2
df = pd.read_excel('mf796-hw5-opt-data.xlsx')

df_call = df[['expDays', 'expT', 'K', 'call_bid', 'call_ask']]
df_put = df[['expDays', 'expT', 'K', 'put_bid', 'put_ask']]
df_call['mid_price'] = (df_call['call_ask'] + df_call['call_bid']) / 2
df_put['mid_price'] = (df_put['put_ask'] + df_put['put_bid']) / 2
df_call['spread'] = df_call['call_ask'] - df_call['call_bid']
df_put['spread'] = df_put['put_ask'] - df_put['put_bid']
#
print('Are call option prices monotonically decreasing in strike?\n',
      check_monotone(df_call))

print('Are put option prices monotonically increasing in strike?\n',
      check_monotone(df_put, 'put'))

print('Is rate of call option price change to strike price change between (-1, 0)?\n', check_rate(df_call))

print('Is rate of put option price changes to strike price changes between (0, 1)?\n', check_rate(df_put, 'put'))

print('Are all call and put prices convex with respect to changes in strike?\n',
      any(check_convexity(df_call)) and any(check_convexity(df_put)))

alpha=1.5
n=12
B=1000
args = (df_call, df_put, alpha, n, B)

warnings.filterwarnings('ignore')

times = 1
x1 = [1, 0.5, 0.5, -1, 0.1]
bnds1 = ((0.1, 5), (0, 2), (0, 2), (-1, 1), (0, 1))
res1 = minimize(fun, np.array(x1), args=args, method='SLSQP', bounds=bnds1, callback=callbackF)
print(res1.success, res1.fun, res1.x)

times = 1
x2 = [2, 0.2, 0.5, -1, 0.1]
bnds2 = ((0.1, 3), (0, 2), (0, 2), (-1, 0), (0, 1))
res2 = minimize(fun, np.array(x2), args=args, method='SLSQP', bounds=bnds2, callback=callbackF)
print(res2.success, res2.fun, res2.x)

times = 1
x3 = [3, 0.1, 1.5, -0.5, 0.1]
bnds3 = ((0.01, 5), (0, 2), (0, 2), (-1, 0), (0, 1))
res3 = minimize(fun, np.array(x3), args=args, method='SLSQP', bounds=bnds3, callback=callbackF)
print(res3.success, res3.fun, res3.x)

times = 1
x4 = [4, 0.05, 1.5, -0.5, 0.1]
bnds4 = ((2, 5), (0, 1), (1, 2), (-1, 0), (0, 0.5))
res4 = minimize(fun, np.array(x4), args=args, method='SLSQP', bounds=bnds4, callback=callbackF)
print(res4.success, res4.fun, res4.x)

times = 1
x5 = [4, 0.05, 1.6, -0.8, 0.03]
bnds5 = ((3, 5), (0, 0.1), (1.5, 2), (-1, -0.5), (0, 0.05))
res5 = minimize(fun, np.array(x5), args=args, method='SLSQP', bounds=bnds5, callback=callbackF)
print(res5.success, res5.fun, res5.x)

times = 1
x6 = [2.5, 0.05, 1.6, -0.8, 0.03]
bnds6 = ((0, 3), (0, 0.1), (1.5, 2), (-1, -0.5), (0, 0.05))
res6 = minimize(fun, np.array(x6), args=args, method='SLSQP', bounds=bnds6, callback=callbackF)
print(res6.success, res6.fun, res6.x)

weighted
times = 1
x1 = [3, 0.1, 1, -0.8, 0.1]
bnds1 = ((2, 5), (0, 2), (0, 2), (-1, 1), (0, 1))
res = minimize(fun2, np.array(x1), args=args, method='SLSQP', bounds=bnds1, callback=callbackF2)
print(res.success, res.fun, res.x)

times = 1
x2 = [3, 0.1, 1, -0.8, 0.05]
bnds2 = ((1, 3), (0, 1), (0, 2), (-1, 0), (0, 1))
res = minimize(fun2, np.array(x2), args=args, method='SLSQP', bounds=bnds2, callback=callbackF2)
print(res.success, res.fun, res.x)

times = 1
x3 = [4, 0.2, 1.2, -0.5, 0.05]
bnds3 = ((2, 5), (0, 1.5), (0, 2), (-1, 0), (0, 1))
res = minimize(fun2, np.array(x3), args=args, method='SLSQP', bounds=bnds3, callback=callbackF2)
print(res.success, res.fun, res.x)

times = 1
x4 = [3.5, 0.1, 1, -0.7, 0.03]
bnds4 = ((3, 5), (0, 1), (0.5, 2), (-1, -0.5), (0, 0.1))
res = minimize(fun2, np.array(x4), args=args, method='SLSQP', bounds=bnds4, callback=callbackF2)
print(res.success, res.fun, res.x)

times = 1
x5 = [3.5, 0.1, 1, -0.8, 0.05]
bnds5 = ((2, 5), (0, 1), (0, 2), (-1, 0), (0, 0.5))
res = minimize(fun2, np.array(x5), args=args, method='SLSQP', bounds=bnds5, callback=callbackF2)
print(res.success, res.fun, res.x)

times = 1
x6 = [5, 0.1, 1, -0.8, 0.05]
bnds6 = ((3, 5), (0, 0.5), (1, 2), (-1, -0.5), (0, 0.1))
res = minimize(fun2, np.array(x6), args=args, method='SLSQP', bounds=bnds6, callback=callbackF2)
print(res.success, res.fun, res.x)

# problem 3
params = [3.3157, 0.0543, 1.1593, -0.7761, 0.0338]

delta_Heston = Heston_delta(params)

print('delta calculated by Heston model is:', delta_Heston)

sigma = \
    root(lambda x: BS_model(275, x, 1 / 4, 267.15, 0.015, 0.0177) - FFT(params, 1 / 4).Heston_fft(1.5, 12, 1200, 275),
         0.47).x[0]

delta_bs = BS_delta(sigma)

print('delta calculated by BS model is:', delta_bs)

np.array(params) + np.array([0, 0.01, 0, 0, 0.01])

print('vega calculated by Heston model is:', Heston_vega(params))
print('vega calculated by BS model is:', BS_vega(sigma))
