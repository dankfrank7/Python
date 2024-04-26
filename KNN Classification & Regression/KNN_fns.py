import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.utils import shuffle
from sklearn.datasets._samples_generator import make_regression
from collections import Counter

def normalise(X, minmax_list=None):
    '''Uniform normalisation of feature values'''

    num_features = X.shape[1]
    X_norm = X.copy().astype(float) # take a copy of X so that X remains the same, in the type of a float
    output_minmax = []
    
    # Loop through every feature index (number columns in X)
    for feature_index in range(num_features):

        # Check if there a min-max input
        if minmax_list is None: 
            x_min = min(X[:,feature_index])
            x_max = max(X[:,feature_index])
        else: 
            x_min = minmax_list[feature_index]['min']
            x_max = minmax_list[feature_index]['max']

        # Uniform normalisation
        X_norm[:, feature_index] = (X_norm[:, feature_index] - x_min) / (x_max - x_min)

        output_minmax.append({'min':x_min,'max':x_max})

    return X_norm, output_minmax

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

def classifier(neighbour_arr, Y):
    '''Selects the most common class from the K nearest neighbour array'''

    class_arr = [Y[i[0]] for i in neighbour_arr]
    return Counter(class_arr).most_common(1)[0][0]

def regressor(neighbour_arr, Y):
    '''Regresses new point based on neighbouring points'''

    # Needs to be updated with weightings for distance

    y_arr =  np.array([Y[neighbour[0]] for neighbour in neighbour_arr])
    weights = np.array([1/neighbour[1] for neighbour in neighbour_arr])

    #return np.mean(y_arr)
    return weighted_average(y_arr, weights) # Weighted by distance

def weighted_average(Y, weights):
    '''Returns a weighted average based on distance'''

    return np.sum(Y*weights/np.sum(weights))