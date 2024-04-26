import numpy as np 
import pandas as pd

def calculate_prior(df, Y): 
    '''Caculate prior probabilities and returns series of class probabilites (prior)'''

    # 1) Find all types of classes
    classes = sorted(list(df[Y].unique()))
    prior = []

    # 2) Loop through each class and calcualte the probabilities
    for i in classes: 
        prior.append(len(df[df[Y]==i]) / len(df))

    return prior 

def calculate_prior2(df: pd.DataFrame, Y: str) -> pd.Series: 
    '''A better and faster version of calculate_prior()'''

    if Y not in df.columns: 
        raise ValueError(f"Column '{Y}' not found in the DataFrame.")

    class_counts = df[Y].value_counts(normalize=True)
    return class_counts.sort_index()


def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label): 
    '''Aproach 1 Calculate P(X=x|Y=y) using Gaussian dist'''
    
    # Split the dataframe into where Y = y
    df_givenY = df[df[Y]==label]
    
    # Calculaate mean and standard deviation of the feature for the adjusted datatset
    mean, std = df_givenY[feat_name].mean(), df_givenY[feat_name].std()

    # Calculate the probability assuming gaussian distribution
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2)))

    return p_x_given_y 

def naive_bayes_gaussian(df, X, Y): 
    '''GAUSSIAN: Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum '''

    # Prior probability
    prior = calculate_prior(df, Y)

    # get feature names
    features = list(df.columns)[:-1]

    Y_pred = []

    # Loop over every test data sample
    for x in X: 

        # Calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        
        # For every unique Y class/label...
        for y in range(len(labels)):

            # Loop through every feature...
            for i in range(len(features)): 

                # And prod the gaussian probabilities for each feature
                likelihood[y] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[y])

        # Caculate posterior probability 
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        # Extract the maximum out 
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label): 
    '''Approach 2: Calculate P(X=x|Y=y) categorically'''

    df_givenY = df[df[Y] == label]
    p_x_given_y = len(df_givenY[df_givenY[feat_name]==feat_val]) / len(df_givenY)

    return p_x_given_y

def naive_bayes_categorical(df, X, Y): 
    '''CATEGORICAL: Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum '''

    # Extract list of features
    features = list(df.columns)[:-1]

    # Calculate prior probability 
    prior = calculate_prior(df, Y)

    # Loop through every data point
    Y_pred = []
    for x in X: 

        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)

        # for every class...
        for y in range(len(labels)):

            # Loop through every feature... 
            for i in range(len(features)): 

                # ...and prod the probability of each feature
                likelihood[y] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[y])

        post_prob = [1]*len(labels)

        # Loop through every class...
        for j in range(len(labels)):

            # ...and calculate the overall post probability
            post_prob[j] = likelihood[j] * prior[j]

        # Extract the index with maximum probability
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)