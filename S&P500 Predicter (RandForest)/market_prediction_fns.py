import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

def predict(train, test, predictors, model, threshold=.6):
    '''This function trains a model and produces predictions'''

    # 1) fit the model to the test data
    model.fit(train[predictors], train["Target"])
    
    # 2) generate predictions from the model
    # Predict proba models 
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0

    # 2.1) reformat predictions into a pd.Series
    preds = pd.Series(preds, index=test.index, name="Predictions")

    # 2.2) combine the actual target variable with predictions from the model
    combined = pd.concat([test["Target"], preds], axis=1)

    return combined 

def backtest(data, model, predictors, start=5000, step=250):
    '''This model back tests the model predictions against historic data'''

    all_predictions = []

    # Loop from the start to the end of the dataset in steps 
    for i in range(start, data.shape[0], step):
        
        # 1) Dynamically slice the train and test data
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        # 2) Generate predictions for the current slice 
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)












