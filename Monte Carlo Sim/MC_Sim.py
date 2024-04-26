"""
Implement the Monte Carlo Method to simulate a stock performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# Get data function
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Define stocks
stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]  # Yahoo requires .AX after every ticker
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

# Get data
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

# Portfolio weights
# weights = np.random.random(len(meanReturns))
# weights /= np.sum(weights)
# Equally weighted portfolio
weights = np.full(len(meanReturns), 1/len(meanReturns))


# Monte caro method
# number of simulations
mc_sims = 500
T = 100 #timeframe in days

# Initialise arrays
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value = 0.0)

initialPortfolio = 10000

# MC Loop
for m in range(0,mc_sims): 
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1)*initialPortfolio

# Plotting figure 1, monte carlo simulation of a stock portfolio
plt.close(1)
plt.figure(1)
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC Simulation of a Stock Portfolio')

# Plotting Figure 2: Probability Distribution of Returns
final_portfolio_values = portfolio_sims[-1, :]
returns_percentage = (final_portfolio_values / initialPortfolio - 1) * 100
mean_return = np.mean(returns_percentage)
std_deviation = np.std(returns_percentage)
plt.close(2)
plt.figure(2)
plt.hist(returns_percentage, bins=30, edgecolor='black')
plt.xlabel('Returns (%)')
plt.ylabel('Frequency')
plt.title('Probability Distribution of Returns')
plt.axvline(x=std_deviation + mean_return, color='red', linestyle='dashed', linewidth=2, label='Standard Deviation')
plt.axvline(x=-std_deviation + mean_return, color='red', linestyle='dashed', linewidth=2,)
plt.axvline(x= mean_return, color='black', linestyle='dashed', linewidth=2,label ='Mean Return')
plt.legend()

print(f"Standard Deviation of Returns: {std_deviation:.2f}%")
print(f"Mean Returns: {mean_return:.2f}%")

plt.show()