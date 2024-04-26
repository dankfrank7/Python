from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.metrics import r2_score
from sklearn import datasets 
from ada_boost import AdaBoost

# Bespoke unctions
import gradient_boost_fns as gradient_boost

df = pd.read_csv('wine.txt', delimiter=';')
X, y = df[:,:-1], df[:,-1]
X, y = pd.DataFrame(X), pd.Series(y)

#df = datasets.load_diabetes()
#X, y = df.data, df.target
#X, y = pd.DataFrame(X), pd.Series(y)

#train, test = train_test_split(df_norm, test_size=.1, random_state=1)
#X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
#X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]

params = {
    "n_estimators": 1000,
    "max_depth": 2,
    "min_samples_split": 2,
    "learning_rate": 0.01,
    "max_features":'sqrt',
    "loss": "squared_error",
}

seeds = [36,1563,12,632,7,2126,878,326,8542,62334]
acc_vec = []
for seed in seeds:

    #model = AdaBoost(n_clf=20) Doesn't work yet needs debugging, something wrong with passing pd.DataFrame and pd.Series
    model = GradientBoostingRegressor(**params) # SKLEARN 37% accuracte
    #model = gradient_boost.GradientBoostRegressor(num_trees=1000, max_depth=1, min_samples_split=2, learning_rate=0.001) #34 % accurate
    #model =  HistGradientBoostingRegressor(    interaction_cst=[[i] for i in range(X.shape[1])], random_state=0) # 36% accurate
    #model = DecisionTreeRegressor(max_depth=4, min_samples_split=2) # 30% Accurate
    #model = RandomForestRegressor(max_depth=4, min_samples_split=2, random_state=2, n_estimators=500) #40% accurate

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    acc = r2_score(Y_test, Y_predict)
    acc_vec.append(acc)

overall_acc = np.mean(acc_vec)
print(overall_acc)
