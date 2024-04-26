import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import pandas as pd 
import Kmeans_fns as kmean

# To Do 
# - Make it able to handle k clusters
# - Make it able to handle n-dimensions
# - Test k vs residual
# - Test other distance computations (manhatten)

## Prepare dataset
# Load data 
X_train, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=20)

# Visualise data
df = pd.DataFrame(dict(x=X_train[:,0], y=X_train[:,1]))
fig, ax = plt.subplots(figsize=(8,8))
df.plot(ax=ax, kind='scatter',x='x',y='y')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show(block=False)

## Test the algorithm
kMax = 16
J_vec = []
for k in range(1,kMax):
    cluster, centroid = kmean.k_means(k, X_train, plot=False)
    J = kmean.cost_function(cluster, centroid, X_train)
    J_vec.append(J)

plt.figure()
plt.plot(list(range(1,kMax)), J_vec, marker='x')
plt.xlabel('k')
plt.ylabel('J')
plt.grid(1)
plt.show(block=False)

input('Press Enter to close figures...')