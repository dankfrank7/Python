"""
Implementation of Modern Portfolio Theory
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go

# Get data function
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

# Portfolio performance 
def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]

def portfolioReturn(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[0]

# Negative sharpe ratio
def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturns-riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1)):
    "Minimise the negative SR by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1}) # constraint that weights add up to 100%
    bound = constraintSet 
    bounds = tuple(bound for asset in range(numAssets)) # create a tuple of bounds for each asset weight
    result = sc.minimize(negativeSR,
                         numAssets*[1./numAssets],
                         args=args,
                         method = 'SLSQP',
                         bounds = bounds, 
                         constraints = constraints)
    return result

def minimizeVariance(meanReturns, covMatrix, riskFreeRate = 0, constraintSet = (0,1)):
    "Minimize the portfolio variance by altering the weights of assets in the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun':lambda x: np.sum(x) - 1}) # constraint that weights add up to 100%
    bound = constraintSet 
    bounds = tuple(bound for asset in range(numAssets)) # create a tuple of bounds for each asset weight
    result = sc.minimize(portfolioVariance,
                         numAssets*[1./numAssets],
                         args=args,
                         method = 'SLSQP',
                         bounds = bounds,
                         constraints = constraints)
    return result

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet = (0,1)):
    """For each return target, we want to optimise the portfolio for min variance """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq','fun':lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                   {'type':'eq', 'fun':lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, 
                        numAssets*[1./numAssets], 
                        args = args, 
                        constraints = constraints,
                        method = 'SLSQP', 
                        bounds = bounds)
    return effOpt

def calculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constraintSet =(0,1)):
    """Read in mean, cov matrix and other financial information
    output, Max SR, min Volatility, efficient frontier """

    # Max sharpe ratio portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index = meanReturns.index, columns = ['Weights'])
    maxSR_allocation.Weights = [round(i*100,2) for i in maxSR_allocation.Weights]

    # Min volatility portfolio
    minVar_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVar_returns, minVar_std = portfolioPerformance(minVar_Portfolio['x'], meanReturns, covMatrix)

    minVar_allocation = pd.DataFrame(minVar_Portfolio['x'], index = meanReturns.index, columns = ['Weights'])
    minVar_allocation.Weights = [round(i*100,2) for i in minVar_allocation.Weights]

    # Efficient frontier
    efficientList = []
    targetReturns = np.linspace(minVar_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])

    # Round outputs
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVar_returns, minVar_std = round(minVar_returns*100,2), round(minVar_std*100,2)

    return maxSR_returns, maxSR_std, maxSR_allocation, minVar_returns, minVar_std, minVar_allocation, efficientList, targetReturns

def EF_graph(meanReturns, covMatrix, riskFreeRate = 0, constraintSet =(0,1)):
    """Return a graph plotting the min var, max sr and efficient frontier"""
    # Call results function
    maxSR_returns,maxSR_std, maxSR_allocation, minVar_returns, minVar_std, minVar_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix)

    # Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3,color='black'))
    )

    # Min Var
    MinVariance = go.Scatter(
        name='Minimum Variance',
        mode='markers',
        x=[minVar_std],
        y=[minVar_returns],
        marker=dict(color='green',size=14,line=dict(width=3,color='black'))
    )

    # Efficient frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100,2) for ef_std in efficientList],
        y=[round(target*100,2) for target in targetReturns],
        marker=dict(color='black',size=4,symbol='circle')
    )

    data = [MaxSharpeRatio, MinVariance, EF_curve]

    # figure layout 
    layout = go.Layout(
        title= 'Portfolio Optimisation',
        yaxis= dict(title='Annualised Return (%)'),
        xaxis= dict(title='Annualised Volatility (std) (%)'),
        showlegend= True,
        legend= dict(
            x=0.75,
            y=0,
            traceorder= 'normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2,
        ), 
        width=800,
        height=600
    )

    # Create figure
    fig = go.Figure(data=data, layout=layout)

    return fig.show()

# Define stocks
#stockList = ['IVV', 'VGS', 'VDHG', 'DRO'] # toms portfolio
stockList = ['IOZ','SYI','GRNV','IOO','DHHF','IEM','IXJ','ETHI','NDQ'] # james portfolio
stocks = [stock + '.AX' for stock in stockList]  # Yahoo requires .AX after every ticker
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)

# Get data
meanReturns, covMatrix = get_data(stocks, startDate, endDate)

EF_graph(meanReturns, covMatrix)

maxSR_returns,maxSR_std, maxSR_allocation, minVar_returns, minVar_std, minVar_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix)

print("Minimum Variance")
print(minVar_allocation)
print("Max Sharpe Ratio")
print(maxSR_allocation)

# Extract weights and stock symbols
minVar_weights = minVar_allocation['Weights'].values
minVar_symbols = minVar_allocation.index

maxSR_weights = maxSR_allocation['Weights'].values
maxSR_symbols = maxSR_allocation.index

# Plot pie chart for Minimum Variance portfolio
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(minVar_weights, labels=minVar_symbols, autopct='%1.1f%%', startangle=140)
plt.title('Minimum Variance Portfolio')

# Plot pie chart for Maximum Sharpe Ratio portfolio
plt.subplot(1, 2, 2)
plt.pie(maxSR_weights, labels=maxSR_symbols, autopct='%1.1f%%', startangle=140)
plt.title('Maximum Sharpe Ratio Portfolio')

# Show the plot
plt.show()