import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from decicion_tree_classification import DecisionTreeClassifier

def main():
    '''Decision tree clasification mainf file'''

    col_names = ['sepal_length','sepal_width', 'petal_length','petal_width','type']
    df = pd.read_csv("iris.csv", skiprows=1,header=None,names=col_names)

    # Train-Test split
    # This code splits the dataset into a train and test data by using a random number generator. 
    # We have selected 20% to be used as test whilst the remaining 80% will be used for training
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

    # Fit the model
    classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
    classifier.fit(X_train,Y_train)

    # Test the accuracty
    Y_pred = classifier.predict(X_test) 
    print('Accuracy: ',round(accuracy_score(Y_test, Y_pred)*100,2),'%')

    # Plot decision tree
    classifier.print_tree()
    classifier.plot_decision_tree()

if os.path.basename(__file__) == '__main__.py':
    main()
