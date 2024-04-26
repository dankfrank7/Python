import yfinance as yf 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import market_prediction_fns as fn 

## Load data 
sp500_df = pd.read_csv('sp500.csv')
# Reset index
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
sp500_df.set_index('Date',inplace=True)
# Delete columns we don't need
del sp500_df["Dividends"]
del sp500_df["Stock Splits"]

#sp500 = yf.Ticker("^GSPC")
#sp500 = sp500.history(period="max")

# Plot price history over time
plot = False
if plot:
    sp500_df.plot.line(y='Close', use_index=True)
    plt.show(block=False)

## Create a target 
sp500_df["Tomorrow"]  = sp500_df["Close"].shift(-1)
sp500_df["Target"]  = (sp500_df["Tomorrow"] > sp500_df["Close"]).astype(int)

# Remove data before 1990 (assuming that market may have changed fundamentally)
sp500_df = sp500_df.loc["1990-01-01":].copy()

## Add in predictors
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons: 

    # Rolling average predictor
    rolling_averages = sp500_df.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500_df[ratio_column] = sp500_df["Close"] / rolling_averages["Close"]

    # Trend predictor
    trend_column = f"Trend_{horizon}"
    sp500_df[trend_column] = sp500_df.shift(1).rolling(horizon).sum()['Target']

    new_predictors += [ratio_column, trend_column]

# Remove rows with NaN
sp500_df = sp500_df.dropna()

# Preditors
old_predictors = ["Close", "Volume", "Open", "High", "Low"]
predictors = old_predictors + new_predictors

## Load RandomForest model
model = RandomForestClassifier(n_estimators=300, min_samples_split=50, random_state=1)

## Test the model
predictions = fn.backtest(sp500_df, model, predictors)

# Calculate overall precision
p = precision_score(predictions["Target"], predictions["Predictions"])
print('Overall Precision: ',round(p*100,2),'%')         

# Calculate precicison of Positive predictions 
pos_predictions = predictions[predictions["Predictions"]==1]
p_pos = precision_score(pos_predictions["Target"], pos_predictions["Predictions"])
print('Precision on pos+ prediction: ',round(p_pos*100,2),'%')

                         

if plot: 
    print(sp500_df.tail(10))
    input('Press enter to close figures...')