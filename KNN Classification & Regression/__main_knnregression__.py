import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets._samples_generator import make_regression
import KNN_fns as knn

def main():

    ## Prepare data
    # Get regression data
    X_train, Y_train = make_regression(n_samples=300, n_features=2, n_informative=2, noise=5, bias=200, random_state=200)

    # Normalise datsets
    X_train_n, minmax_list = knn.normalise(X_train)

    # New datapoints
    new_points = np.array([[1, 1],
                        [0.8, .2],
                        [0.5, 0.5],
                        [0.2, 0.8],
                        [0, 0]])
   
    # Normalise new datapoints
    new_points_n, _ = knn.normalise(new_points)

    ## Regression
    # Run the KNN Regression
    n = 5
    predictions = []
    neighbours = []
    # Loop through every new point
    for point in new_points_n:
        knn_reg = knn.find_neighbours(n, X_train_n, point)
        y_predict = knn.regressor(knn_reg, Y_train)
        predictions.append(y_predict)
        neighbours.append(knn_reg)

    # Convert to np.arr
    neighbours = np.array(neighbours)
    indexes = neighbours[:,:,0].reshape(-1).astype(int)
    
    ## Plotting 
    # Training data plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(X_train_n[:,0],X_train_n[:,1], Y_train, c = 'gray' ,alpha=.5, marker = 'o')
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')

    # Plot regressed points
    plt.plot(new_points_n[:,0], new_points_n[:,1], predictions ,linestyle = '', marker='o', c='red')
    plt.plot(X_train_n[indexes,0], X_train_n[indexes,1], Y_train[indexes] ,linestyle = '', marker='o', c='orange')
    plt.show()

if __name__ =="__main__":
    print("Running...")
    main()


  