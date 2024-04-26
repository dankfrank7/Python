import numpy as np 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
import pandas as pd 

## Prepare dataset
# Load data 
X_train, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=20)

# Visualise data
#df = pd.DataFrame(dict(x=X_train[:,0], y=X_train[:,1]))
#fig, ax = plt.subplots(figsize=(8,8))
#df.plot(ax=ax, kind='scatter',x='x',y='y')
#plt.xlabel('X_1')
#plt.ylabel('X_2')
#plt.show()

## Helper functions 
def init_centroids(k, X):
    '''Creates first guess of centroids for k centers with dataset X of n dimensions'''

    # To be updated to handle n dimensions

    arr = []
    for i in range(k): 
        cx1 = np.random.uniform(min(X[:,0]), max(X[:,0]))
        cx2 = np.random.uniform(max(X[:,0]), max(X[:,0]))
        arr.append([cx1, cx2])

    return np.asarray(arr)

def dist(a, b):
    '''Calculates euclidian distasnce between points a and b'''

    return np.sqrt(sum(np.square(a-b)))

def assign_cluster(k, X, cg):
    '''Assigns a custer class to every point in X, based on the nearest centroid'''

    # Make initial list of -1s 
    clusters = [-1]*len(X)

    # Loop through every data point
    for i in range(len(X)): 
        dist_arr = []

        # Loop through every cluster
        for j in range(k): 
            dist_arr.append(dist(X[i], cg[j]))
        
        # Find smallest distance 
        idx = np.argmin(dist_arr)
        clusters[i] = idx

    return np.asarray(clusters)

def compute_centroids(k, X, cluster):
    '''Computes the centroid of classes 'k' in dataset X '''

    cg_arr = []
    
    # Loop through every cluster
    for i in range(k):
        arr = []

        # Loop over every datapoint
        for j in range(len(X)):
            if cluster[j] == i:
                arr.append(X[j])
        
        # Centre of gravity is mean of dimensions
        cg_arr.append(np.mean(arr, axis=0))

    return np.asarray(cg_arr)

def measure_change(cg_prev, cg_new): 
    '''Calculates the change in centroids'''
    res = 0
    for a,b in zip(cg_prev, cg_new):
        res += dist(a,b)

    return res

def show_clusters(X, cluster, cg): 
    '''Plots clusters'''

    df = pd.DataFrame(dict(x=X_train[:,0], y=X_train[:,1], label=cluster))
    colours = {0:'blue', 1:'orange', 2:'green'}
    fig, ax = plt.subplots(figsize=(8,8))
    grouped = df.groupby('label')
    
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,color=colours[key])
    
    ax.scatter(cg[:,0], cg[:,1], marker='*', s=150, c='#ff2222')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

def k_means(k, X, threshold=0.001, plot=True):
    '''Run the k means clustering algorithm for k clusters'''

    cg_prev = init_centroids(k, X)
    cluster = [0]*len(X)
    cg_change = 100

    # Loop while error is larger than threshold 
    while cg_change > threshold:

        # 1) Assign Cluster
        cluster = assign_cluster(k, X, cg_prev)

        # (optional) plot clusters        
        if plot:
            show_clusters(X, cluster, cg_prev)

        # 2) Calculate new centroids
        cg_new = compute_centroids(k, X, cluster)

        # 3) Calculate change in centroids
        cg_change = measure_change(cg_new, cg_prev)

        # 4) Update centroids
        cg_prev = cg_new
        
    return cluster


# Test the algorithm
k = 3
cluster = k_means(3, X_train)