import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

def init_centroids(k, X):
    '''Creates first guess of centroids for k centers with dataset X of n dimensions'''

    # To be updated to handle n dimensions

    arr = []
    for i in range(k): 
        cx1 = np.random.uniform(min(X[:,0]), max(X[:,0]))
        cx2 = np.random.uniform(min(X[:,1]), max(X[:,1]))
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

    # If a centroid is not classifying any points, then randomly assign it somewhere
    for i, cg in enumerate(cg_arr):
        if np.any(np.isnan(cg)):
            cg_arr[i] = init_centroids(1,X).reshape(-1)
        else:
            cg.reshape(-1)
    
    return np.asarray(cg_arr)

def measure_change(cg_prev, cg_new): 
    '''Calculates the change in centroids'''
    res = 0
    for a,b in zip(cg_prev, cg_new):
        res += dist(a,b)

    return res

def show_clusters(X, cluster, cg): 
    '''Plots clusters'''

    df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=cluster))
    num_clusters = len(np.unique(cluster))
    cmap = plt.get_cmap('RdYlBu')
    fig, ax = plt.subplots(figsize=(8,8))
    grouped = df.groupby('label')
    
    for key, group in grouped:
        color = cmap(key / num_clusters)
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=color)
    
    ax.scatter(cg[:,0], cg[:,1], marker='*', s=150, c='#ff2222')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show(block=False)

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
        
    return cluster, cg_new

def cost_function(cluster, centroid, X):
    '''Calcualtes the cost function J for the K-Means clustering alogrithm'''

    res = 0
    for i, cg in enumerate(centroid):
        for j, x in enumerate(X):
            if cluster[j] == i:
                res += np.sum(np.square(x - cg))

    return res