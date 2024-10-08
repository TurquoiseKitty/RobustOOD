'''randomly generated prices and budget, where each component of the price vector is a random number between 0 and 1'''

import numpy as np
import pandas as pd

def generate_random_prices_and_budget(dim_c,num_samples,random_seed):
    np.random.seed(random_seed)
    # uniformly choose integer prices between 1 and 1000
    prices = np.random.randint(low=1,high=1000,size=(num_samples,dim_c))
    
    low = np.max(prices,axis=1)
    low = low.reshape((num_samples,1))
    high = prices @ np.ones((dim_c,1))
    high = high - low*np.random.uniform(size=(num_samples,1))
    budgets = np.random.uniform(low=low,high=high)
    return prices,budgets



if __name__ == '__main__':
    random_seed = 0
    dim_c = 20
    num_samples = 10
    prices,budgets = generate_random_prices_and_budget(dim_c,num_samples,random_seed)
    # save to csv without headers and index
    pd.DataFrame(prices).to_csv("prices.csv",header=False,index=False)
    pd.DataFrame(budgets).to_csv("budgets.csv",header=False,index=False)