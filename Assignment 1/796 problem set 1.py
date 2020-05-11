
"""
Problem set 1
@author: zehaodong
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

class Call_option:
    def __init__(self, r, S0, K, sigma, T, beta):
        self.r = r
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.beta = beta
    
    def simulation(self, mean, variance):
        '''
        Simulation for only one path
        '''
        w = np.random.normal(mean, variance ** 0.5, size=self.T * 252)
        result = []
        S = self.S0
        result.append(S)
        for c in w:
            S = self.r * S * (self.T / 252) + self.sigma * S ** self.beta * c + S
            result.append(S)
        return pd.DataFrame(result)
    
    def simulation_paths(self, paths, mean, variance):
        '''
        Simulation for a number of paths, 
        the number determines by the variable 'paths'
        '''
        result = pd.DataFrame()
        while paths > 0:
            a = self.simulation(mean, variance)
            result = pd.concat([result, a], axis = 1)
            paths -= 1
        return result
    
    def simulation_price(self, df):
        '''
        Terminal value
        '''
        value = df.iloc[-1, :]
        return value.reset_index(drop = True)
    
    def discount(self, price):
        '''
        Dicount factor for continuos compounding
        '''
        D = np.exp(-self.r * self.T)
        price = price * D
        return price

    def BSformula(self):
        '''
        Price calculated by Black-Scholes formula
        '''
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        price = self.S0 * Nd1 - Nd2 * self.K * np.exp(-self.r * self.T)
        return price
    
    def delta(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * self.T ** 0.5)
        delta = stats.norm.cdf(d1)
        return delta
    
    def payoff(self, terminal_price):
        result = []
        for c in terminal_price:
            if c > self.K:
                result.append(c - self.K)
            else:
                result.append(0)
        return pd.Series(result)
    
    def price(self, terminal_price):
        return self.discount(np.mean(self.payoff(terminal_price)))
    
    def delta_payoff(self, terminal_price):
        delta = self.delta()
        payoff = np.mean(self.discount(self.payoff(terminal_price)) - delta * self.discount(terminal_price - self.S0))
        return payoff
    
if __name__ == "__main__":    
#(b)
    call = Call_option(0, 100, 100, 0.25, 1, 1)
    sim_result = call.simulation_paths(1000, 0, 1 / 252)
    terminal_price = call.simulation_price(sim_result)
    price = call.price(terminal_price)
    print('the price of an at-the-money one year European call option is', price)
#(c)
    price_BS = call.BSformula()
    print('the price of the European call option calculated via formula is', price_BS)
#(d)(e)
    delta = call.delta()
    print('the delta of the European call option is', delta)
#(f)
    payoff = call.delta_payoff(terminal_price)
    print('the payoff of the delta neutral portfolio is', payoff)
    print('upper bound', (100 - delta * 100 + price_BS) / (1 - delta))
    print('lower bound', (delta * 100 - price_BS) / delta)
#(g)
    call2 = Call_option(0, 100, 100, 0.25, 1, 0.5)
    sim_result2 = call2.simulation_paths(1000, 0, 1 / 252)
    terminal_price2 = call2.simulation_price(sim_result2)
    payoff2 = call2.delta_payoff(terminal_price2)
    print('the payoff of the delta neutral portfolio(beta=0.5) is', payoff2)
#(h)
    call3 = Call_option(0, 100, 100, 0.4, 1, 1)
    sim_result3 = call3.simulation_paths(1000, 0, 1 / 252)
    terminal_price3 = call3.simulation_price(sim_result3)
    payoff3 = call3.delta_payoff(terminal_price3)
    print('the payoff of the delta neutral portfolio(sigma=0.4) is', payoff3)