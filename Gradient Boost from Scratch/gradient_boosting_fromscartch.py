from matplotlib import pyplot as plt 
from sklearn import datasets 
from sklearn.tree import DecisionTreeRegressor 
from sklearn import tree 
import pandas as pd 
import numpy as np 
import graphviz 
import re
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, explained_variance_score

# Functions
import gradient_boost_fns as gradient_boost

#data = datasets.load_diabetes(as_frame=True, scaled=True)
#df = data.frame

df = pd.read_csv('wine.txt', delimiter=';')

# Normalise features and target
df_norm= gradient_boost.normalise(df)
train, test = train_test_split(df_norm, test_size=.1, random_state=41)

X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]

# Plot distribution of the features
gradient_boost.feature_hists(X_train)

model = gradient_boost.GradientBoostRegressor(num_trees=100, max_depth=3, min_samples_split=5)
model.fit(X_train, Y_train)
model.plot_residuals(log_scale=True)

# Test model 
#Y_predict = model.predict(X_test)
#r2 = r2_score(Y_test, Y_predict)
#print(f'R2 Score: {round(r2,3)}')

#Y_test_vs_predict = pd.concat([Y_test, Y_predict],axis=1,keys=['Y_test','Y_predict'])

accuracy = model.tune(X_train, Y_train, X_test, Y_test, params=None)

plt.figure()
plt.contourf(accuracy)  # Create filled contour plot
plt.colorbar()        # Add color bar (optional)
plt.xlabel('max_depth')
plt.ylabel('num_trees')
plt.show(block=False)   


input('Press enter to close figures...')