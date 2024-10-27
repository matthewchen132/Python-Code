import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

# Import data (pandas_datareader)
def getData(stocks, start, end):
    try:
        stockData = yf.download(stocks, start=start, end=end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return returns, meanReturns, covMatrix
    except Exception as e:
        print(f"Error retrieving stock data using pandas_datareader: {e} ")
        print("Using yfinance library...")

stockList = ['AAPL', 'SPY', 'MSFT', 'BTC-USD', 'MA', 'TXRH']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365)
returns, meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
print(meanReturns)
weights = np.random.random(len(stockList))
weights /= weights.sum() # random distribution of stocks
print(f"Stocks: {stockList}")
print(f"Stock weights: {weights}", "\n")


# Now, Monte Carlo!
n_sims = 100
T_days = 100

meanM = np.full(shape=(T_days, len(stockList)), fill_value=meanReturns) # 100 x 5
meanM = meanM.T # transposes 5 x 100 matrix of means

portfolio_sims = np.full(shape=(T_days, n_sims ), fill_value=0.0) # 100 x 100 matrix

initialPortfolioValue = 20000
breakEven = initialPortfolioValue
results = []

for sim in range(0,n_sims):
    # Cholesky decomposition: Rt = mean + LZ^T
        # L is the lower triangular matrix
        # Z transpose is the samples of a normal distribution
    Z = np.random.normal(size=(T_days,len(stockList))) # 100, 5 , already transposed
    L = np.linalg.cholesky(covMatrix) # Covariance = L*L.T
    dailyReturns = meanM + np.inner(L,Z) # dailyReturns over T_days = mean + LZ^T
    portfolio_sims[:,sim] = np.cumprod(np.inner(weights,dailyReturns.T)+1)*initialPortfolioValue # accumulates product of daily returns, adds them and keeps compiling each day
    index_highest_concentration = np.argmax(weights)
    results.append({'Sim #': sim, 'Highest Weight Stock': stockList[index_highest_concentration], 'Total Gain (%)': portfolio_sims[T_days-1,sim]/initialPortfolioValue})



    
plt.plot(portfolio_sims,)
plt.xlabel(f"Stock price of past 100 days from {dt.datetime.now()}")
plt.ylabel("Stock price ($)")
plt.title("Monte Carlo Stock simulation")
plt.show()

print(portfolio_sims[T_days-1,:])
print('RESULTS: ', '\n' , results)


# Some ideas:
# Different strategies? say I had 50% SPY, 50% BTC
# Additionaly functionality
# Accurate to my own 
# Selling and buying on different days

## what are features I can add to be useful
# (1) standard deviations instead of normal distribution
            # stdDevs = returns.std()  # Calculate the standard deviation for each stock

            # # Generate Z using the specified standard deviations
            # Z = np.random.normal(size=(T_days, len(stockList)))  # Standard normal variables

            # # Scale Z by the standard deviations
            # dailyReturns = meanM + np.inner(L, Z * stdDevs.values)


# Additional Features to Consider
# If you're looking to enhance your simulation, consider adding:

# Dynamic Weights: Implement strategies that adjust weights based on performance or other criteria over time.
# Transaction Costs: Introduce transaction costs for buying and selling stocks.
# Different Strategies: Implement various trading strategies, like momentum or mean reversion.
# Visualization Improvements: Add histograms or heatmaps to visualize distributions of returns.
# Risk Metrics: Calculate and visualize risk metrics (e.g., Sharpe ratio, VaR).
# Realistic Scenarios: Use different distributions for returns, like fat tails or skewed distributions, to simulate market behaviors more realistically.
