import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn.datasets._samples_generator import make_regression
from collections import Counter

# Regression plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_reg[:,0],X_train_reg[:,1],Y_train_reg, c = "red",alpha=.5, marker = 'o')
ax.set_xlabel('X_1')
ax.set_ylabel('X_2')
ax.set_zlabel('Y')

# Regression 
new_points_reg = np.array([[-1, 1],
                      [0, 2],
                      [-3, -2],
                      [3, -3]])
new_points_reg_norm, minmax_list = normalise(new_points_reg)

X_train_reg_norm, minmax_list = normalise(X_train_reg)

knn_reg = find_neighbours(3, X_train_reg_norm, new_points_reg_norm[1])
y_predict = KNN_regressor(knn_reg, Y_train_reg)
plt.plot(new_points_reg[0,1], new_points_reg[1,1], y_predict,marker='x', c='green')
plt.show()