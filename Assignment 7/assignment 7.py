import numpy as np
import cmath
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn')


class FFT:
    def __init__(self, params, T, S0=282, r=0.015, q=0.0177):
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


class Simulation:
    def __init__(self, params, steps, N, T, S0=282, r=0.015, q=0.0177):
        self.kappa = params[0]
        self.theta = params[1]
        self.sigma = params[2]
        self.rho = params[3]
        self.v0 = params[4]

        self.steps = steps
        self.N = N
        self.dt = T / steps

        self.T = T
        self.S0 = S0
        self.r = r
        self.q = q

    def sim_paths(self, seed=None):
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.v0
        steps = self.steps
        N = self.N
        dt = self.dt
        S0 = self.S0
        r = self.r
        q = self.q
        T = self.T

        mean = (0, 0)
        cov = [[dt, rho * dt], [rho * dt, dt]]
        if seed:
            np.random.seed(seed)
        W = np.random.multivariate_normal(mean, cov, (N, steps))
        W_S = W[:, :, 0]
        W_v = W[:, :, 1]
        vt = np.zeros((N, steps))
        vt[:, 0] = v0
        St = np.zeros((N, steps))
        St[:, 0] = S0
        for t in range(1, steps):
            St[:, t] = St[:, t - 1] + (r - q) * St[:, t - 1] * dt + np.sqrt(vt[:, t - 1].clip(min=0)) * St[:,
                                                                                                        t - 1] * W_S[:,
                                                                                                                 t - 1]
            vt[:, t] = vt[:, t - 1] + kappa * (theta - vt[:, t - 1]) * dt + sigma * np.sqrt(
                vt[:, t - 1].clip(min=0)) * W_v[:, t - 1]

        return St

    def payoff_euro(self, K, seed=None):
        St = self.sim_paths(seed)
        result = (St[:, -1] - K).clip(min=0)

        return result * np.exp(-self.r * self.T)

    def payoff_up_out(self, K1, K2, seed=None):
        St = self.sim_paths(seed)
        max_S = np.max(St, axis=1)
        result = (St[:, -1] - K1).clip(min=0)
        result[max_S >= K2] = 0

        return result * np.exp(-self.r * self.T)

    def payoff_control_variate(self, K1, K2, mu, seed=None):
        St = self.sim_paths(seed)
        max_S = np.max(St, axis=1)
        payoff_euro = (St[:, -1] - K1).clip(min=0)
        payoff_up_out = payoff_euro.copy()
        payoff_up_out[max_S >= K2] = 0

        cov = np.cov(np.array([payoff_euro, payoff_up_out]))[0][1]
        var = np.cov(np.array([payoff_euro, payoff_up_out]))[0][0]
        c = -cov / var
        theta = payoff_up_out + c * (payoff_euro - mu)

        return theta * np.exp(-self.r * self.T)


if __name__ == '__main__':
    M = 252
    T = 1
    N = 10000
    S0 = 282
    q = 1.77 / 100
    r = 1.5 / 100
    params = (3.32, 0.054, 1.16, -0.78, 0.034)
    K1 = 285
    K2 = 315

    option = Simulation(params, M, N, T)
    # 3.
    alpha = 1.5
    n = 12
    B = 1000
    mu_FFT = FFT(params, 1).Heston_fft(alpha, n, B, 285)
    print('The price of European call calculated by FFT is:', mu_FFT)

    option.N = 150000
    res1 = option.payoff_euro(K1, 1)
    Ns = (100, 500, 1000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000, 150000)
    p_euro1 = []
    for i in Ns:
        p_euro1.append(np.mean(np.random.choice(res1, i)))

    table1 = pd.DataFrame({'N': Ns, 'Price': p_euro1, 'Error':abs(p_euro1 - mu_FFT)})
    print(table1)

    # 4.
    res2 = option.payoff_up_out(K1, K2, 1)
    mu_up_out = np.mean(option.payoff_up_out(K1, K2, 1))

    p_up_out = []
    for i in Ns:
        p_up_out.append(np.mean(np.random.choice(res2, i)))

    table2 = pd.DataFrame({'N': Ns, 'Price': p_up_out, 'Error':abs(p_up_out - mu_up_out)})
    print(table2)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Up-and-out Call Price for Different N')
    ax.set_ylabel('Price')
    ax.set_xlabel('N')
    ax.plot(Ns, p_up_out, "o-")

    # 5.
    option.N = 150000
    res3 = option.payoff_control_variate(K1,K2,mu_FFT,1)

    p_control_variate = []
    for i in Ns:
        p_control_variate.append(np.mean(np.random.choice(res3, i)))

    table3 = pd.DataFrame({'N': Ns, 'Price without cv': p_up_out, 'price with cv': p_control_variate,
                           'Error without cv': abs(p_up_out - mu_up_out), 'Error with cv': abs(p_control_variate - mu_up_out)})
    print(table3)


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Up-and-out Call Price for Different N')
    ax.set_ylabel('Price')
    ax.set_xlabel('N')
    ax.plot(Ns, p_up_out, "o-", Ns, p_control_variate, "o-")
    ax.legend(['Without control variate', 'With control variate'])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_title('Up-and-out Call Price Error for Different N')
    ax.set_ylabel('Price')
    ax.set_xlabel('N')
    ax.plot(Ns, abs(p_up_out - mu_up_out), "o-", Ns, abs(p_control_variate - mu_up_out), "o-")
    ax.legend(['Without control variate', 'With control variate'])

    St = option.sim_paths(1)
    max_S = np.max(St, axis=1)
    payoff_euro = (St[:, -1] - K1).clip(min=0)
    payoff_up_out = payoff_euro.copy()
    payoff_up_out[max_S >= K2] = 0

    cov = np.cov(np.array([payoff_euro, payoff_up_out]))[0][1]
    var = np.cov(np.array([payoff_euro, payoff_up_out]))[0][0]
    print(cov**2 / var)


