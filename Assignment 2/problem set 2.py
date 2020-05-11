"""
Problem set 2
@author: zehaodong
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf

class Call_option:
    def __init__(self, r, S0, K, sigma, T):
        self.r = r
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        self.d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        
    def BSformula(self):
        '''
        Price calculated by Black-Scholes formula
        '''
        Nd1 = stats.norm.cdf(self.d1)
        Nd2 = stats.norm.cdf(self.d2)
        price = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return price
    
    def density(self, x):
        return (1 / (2 * np.pi) ** 0.5) * np.exp(-x ** 2 / 2)
    
    def left_riemann(self, n):
        b1 = self.d1
        b2 = self.d2
        a = -10
        x1 = np.array([a + (b1 - a) / n * i for i in range(n)])
        v1 = self.density(x1)
        Nd1 = np.sum((b1 - a) / n * v1)
        x2 = np.array([a + (b2 - a) / n * i for i in range(n)])
        v2 = self.density(x2)
        Nd2 = np.sum((b2 - a) / n * v2)
        result = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return result
    
    def midpoint(self, n):
        b1 = self.d1
        b2 = self.d2
        a = -10
        x1 = np.array([a + (b1 - a) / n * (i + 1/2) for i in range(n)])
        v1 = self.density(x1)
        Nd1 = np.sum((b1 - a) / n * v1)
        x2 = np.array([a + (b2 - a) / n * (i + 1/2) for i in range(n)])
        v2 = self.density(x2)
        Nd2 = np.sum((b2 - a) / n * v2)
        result = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return result
    
    def gauss_legendre(self, n):
        x, w = np.polynomial.legendre.leggauss(n)
        b1 = self.d1
        b2 = self.d2
        a = -10
        v1 = self.density((x * (b1 - a) + (b1 + a)) / 2)
        v2 = self.density((x * (b2 - a) + (b2 + a)) / 2)
        Nd1 = sum(w * v1) * (b1 - a) / 2
        Nd2 = sum(w * v2) * (b2 - a) / 2
        result = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return result
    
    def error(self, x):
        return abs(x - self.BSformula())
    
    def density2(self, x):
        mu = self.S0
        sd = self.sigma
        return 1 / (sd * (2 * np.pi) ** 0.5) * np.exp(-1 * (x - mu) ** 2 / (2 * sd ** 2))
    
    def midpoint2(self, n):
        mu = self.S0
        sd = self.sigma
        b = mu + 10 * sd
        a = self.K
        x = np.array([a + (b - a) / n * (i + 1/2) for i in range(n)])
        v = np.exp(-self.r * self.T) * (x - self.K) * self.density2(x)
        result = np.sum((b - a) / n * v)
        return result

    def density3(self, x, y, sd, rho):
        mu = self.S0
        sd1 = self.sigma
        sd2 = sd
        e = np.exp(-1 / (2 * (1 - rho ** 2)) * ((x - mu) ** 2 / sd1 ** 2 + (y - mu) ** 2 / sd2 ** 2 
                                                - 2 * rho * (x - mu) * (y - mu) / (sd1 * sd2)))
        return (1 / (2 * np.pi * sd1 * sd2 * np.sqrt(1 - rho ** 2))) * e
    
    def contingent(self, sd, K2, n, rho):
        mu = self.S0
        sd1 = self.sigma
        sd2 = sd
        b1 = mu + 10 * sd1
        b2 = K2
        a1 = self.K
        a2 = mu - 10 * sd2
        x1 = np.array([a1 + (b1 - a1) / n * (i + 1 / 2) for i in range(n)])
        x2 = np.array([a2 + (b2 - a2) / n * (i + 1 / 2) for i in range(n)])
        result = 0
        for i in x2:
            v = np.exp(-self.r * self.T) * (x1 - self.K) * self.density3(x1, i, sd, rho)
            result += (b2 - a2) / n * (np.sum((b1 - a1) / n * v))
        return result
    
    
if __name__ == "__main__":   
    # Problem 1
    call = Call_option(0.04, 10, 12, 0.2, 1/4)
    # 1
    print('The price of the call calculated by BS formula is:', call.BSformula())
    # 2 
    nodes = np.array([5, 10, 50, 100])
    left_riemann = [call.left_riemann(i) for i in nodes]
    midpoint = [call.midpoint(i) for i in nodes]
    gauss_legendre = [call.gauss_legendre(i) for i in nodes]
    #(a)
    print('The price calculated by Left Riemann method is:', left_riemann, '\n',
          'The calculation error of Left Riemann method is:', call.error(left_riemann))
    #(b)
    print('The price calculated by Midpoint method is:', midpoint, '\n',
          'The calculation error of Midpoint method is:', call.error(midpoint))
    #(c)
    print('The price calculated by Gauss Legendre method is:', gauss_legendre, '\n',
          'The calculation error of Gauss Legendre method is:', call.error(gauss_legendre))

    # 3
    nodes = np.array([i for i in range(1, 40)])
    left_riemann = [call.left_riemann(i) for i in nodes]
    midpoint = [call.midpoint(i) for i in nodes]
    gauss_legendre = [call.gauss_legendre(i) for i in nodes]
    # figure 1
    plt.figure(figsize=(10,6), dpi=120)
    plt.plot(nodes, call.error(left_riemann))
    plt.plot(nodes, call.error(midpoint))
    plt.plot(nodes, 1 / nodes)
    plt.plot(nodes, 1 / nodes ** 2)
    plt.plot(nodes, 1 / nodes ** 3)
    plt.title('Figure 1. Error of Left-Riemann and Midpoint')
    plt.xlabel('N')
    plt.ylabel('Absolute Value of Error')
    plt.legend(['Left-Riemann', 'Midpoint', 'N^-1', 'N^-2', 'N^-3'])
    plt.show()
    # figure 2
    nodes = np.array([i for i in range(1, 10)])
    
    gauss_legendre = [call.gauss_legendre(i) for i in nodes]
    plt.figure(figsize=(10,6), dpi=120)
    plt.plot(nodes, call.error(gauss_legendre))
    plt.plot(nodes, 1 / nodes ** nodes)
    plt.plot(nodes, 1 / nodes ** (2 * nodes))
    plt.title('Figure 2. Error of Gauss_Legendre')
    plt.xlabel('N')
    plt.ylabel('Absolute Value of Error')
    plt.legend(['Gauss_Legendre', 'N^-N', 'N^-2N'])
    plt.show()
    
    # Problem 2
    SPY = yf.download('SPY', '2020-1-31')['Close'][0]
    call2 = Call_option(0, SPY, 370, 20, 1)
    # 1
    print('The price of the call is:', call2.midpoint2(100))
    # 2
    print('The price of the contingent call is:', call2.contingent(15, 365, 100, 0.95))
    # 3
    print('The price of the contingent calls are:')
    for rho in [0.8, 0.5, 0.2]:
        print(call2.contingent(15, 365, 100, rho))
    # 5 for c in [360, 350, 340]:
    print('The price of the contingent calls are:')
    for c in [360, 350, 340]:
        print(call2.contingent(15, c, 100, 0.95))
    
    
    
    
