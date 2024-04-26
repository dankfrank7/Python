import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets._samples_generator import make_regression
from collections import Counter


# Current bugs
# 1) Regression isn't selecting the correct nearests neighbours

# Load blob dataset
X_train, Y_train = make_blobs(n_samples=300, centers =2, n_features=2, cluster_std=6, random_state=11)
X_train_reg, Y_train_reg = make_regression(n_samples=300, n_features=2, n_informative=2, noise=5, bias=30, random_state=200)

# Load into dataframe
df = pd.DataFrame(dict(x=X_train[:,0], y=X_train[:,1], label=Y_train))

def normalise(X):
    '''Normalise feature values'''

    num_features = X.shape[1]
    minmax_list = []

    for feature_index in range(num_features):
        x_min = min(X[:,feature_index])
        x_max = max(X[:,feature_index])
 
        X[:, feature_index] = (X[:, feature_index] - x_min) / (x_max - x_min)

        minmax_list.append({'min':x_min,'max':x_max})

    return X, minmax_list

def find_neighbours(k, X_train, new_point):
    '''
    Finds k number of neighbours next to the new_point. Only takes one point at a time.
    '''
    
    neighbour_arr = []

    # Loop throught every training data point 
    for i in range(len(X_train)):

        # Calculate the euclidian distance, this should work for different numbers of dimensions (as long as they are identical)
        dist = np.sqrt(sum(np.square(X_train[i]-new_point)))
        neighbour_arr.append([i,dist])

    neighbour_arr = sorted(neighbour_arr, key = lambda x : x[1])

    return neighbour_arr[0:k]

def KNN_classifier(neighbour_arr, Y):
    '''Selects the most common class from the K nearest neighbour array'''

    class_arr = [Y[i[0]] for i in neighbour_arr]
    return Counter(class_arr).most_common(1)[0][0]

def KNN_regressor(neighbour_arr, Y):
    '''Regresses new point based on neighbouring points'''

    y_arr =  [Y[i[0]] for i in neighbour_arr]
    return np.mean(y_arr)

# New point data
new_points = np.array([[-10, -10],
                      [0, 10],
                      [-15, 10],
                      [5, -2]])
new_points_norm, minmax_list = normalise(new_points)

# Normalise data
X_norm, minmax_list = normalise(X_train)

knn = find_neighbours(4, X_norm, new_points[1])
print(KNN_classifier(knn, Y_train))

# Plot dataset
colors = {0:'blue',1:'orange'}
fig, ax = plt.subplots(figsize=(8, 8))
grouped = df.groupby('label')
for key, group in grouped: 
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

plt.plot(new_points[0,1],'o')
plt.xlabel('X_0')
plt.ylabel('X_1')

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