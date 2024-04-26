import pandas as pd
from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from random_forest_classifier import RandomForest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

''' 
To Do: 
- Implement a feature num variable for bootstrapping - DONE
    - Implement a standard n_bs = sqrt(n_features) function - DONE
- Check the identical tree bug
    - Check fit function 
    - Printing / Plotting function
- Check best_split['info gain'] bug DONE
    - Model is sometimes not attributing an ['info_gain'] to 
'''

def accuracy(Y_true, Y_pred):
    accuracy = np.sum(Y_true == Y_pred) / len(Y_true)
    return accuracy

def print_predictions(Y_true, Y_predict, percent_vote):
    for true, predict, vote in zip(Y_true, Y_predict, percent_vote):
        print(f"True: {true}, Predicted: {predict}, % Vote: {round(vote*100,2)}%")


def main(n_trees = 8, max_depth = 3, plot=True):
    '''Random Forest Classifier Main'''

    # Load data
    choose = True
    if choose: 

        data = datasets.load_breast_cancer()
        X = data.data
        Y = data.target.reshape(-1,1)
    else: 
        col_names = ['sepal_length','sepal_width', 'petal_length','petal_width','type']
        df = pd.read_csv("iris.csv", skiprows=1,header=None,names=col_names)
        X = df.iloc[:,:-1].values
        Y = df.iloc[:,-1].values.reshape(-1,1)

    # Split into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    
    # Build random forest model
    forest = RandomForest(n_trees,max_depth)
    forest.fit(X_train, Y_train)
    Y_predict, percent_votes = forest.predict(X_test)

    # Print the accuracy
    acc = accuracy_score(Y_test, Y_predict)
    print('Tree Accuracy: ',round(acc*100,2),'%')

    if plot:
    # Print the predictions 
        print_predictions(Y_test, Y_predict, percent_votes)

        # Plot the forest
        forest.plot_forest()

    return acc

def main_loop(iterations, plot=True): 
    
    acc_vec = []
    
    for i in range(iterations):
        acc = main(plot)
        acc_vec.append(round(acc*100,2))

    plt.figure()
    plt.hist(acc_vec)
    plt.show()

def main_loop2(plot=True): 

    acc_vec = []
    n_tree_vec = [2,4,6,8,10,12,14]
    
    for n in n_tree_vec:
        acc = main(n, plot=False)
        acc_vec.append(round(acc*100,2))

    plt.figure()
    plt.plot(n_tree_vec, acc_vec)
    plt.show()

if __name__ == '__main__':
    main_loop2(plot=False)
    #main(