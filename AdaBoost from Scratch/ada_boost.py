import numpy as np 

class DecisionStump():

    def __init__(self):
        '''Constructor'''
        
        self.polarity = 1
        self.feature_index = None 
        self.threshold = None 
        self.alpha = None 

    def predict(self, X: np.array) -> np.array: 
        '''Takes a np.array of X values and classifies them based on stump threshold and feature index'''

        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions
    
class AdaBoost():

    def __init__(self, n_clf=5):
        '''Constructor'''

        self.n_clf = n_clf 
    
    def fit(self, X, y): 
        '''Fit method'''

        n_samples, n_features = X.shape 

        # Initialise our weights
        weights = np.full(n_samples, (1/n_samples))

        # Loop through each stump 
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()

            # Greedy search over all features...
            min_error = float('inf')
        
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column) 

                # ...and all thresholds
                for threshold in thresholds:

                    # Calculate predictions for current threshold and feature
                    p = 1 
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1 

                    # Calculate the error 
                    misclassified = weights[y != predictions]
                    error = sum(misclassified)

                    # If error is greater than 0.5 we flip the predictions and weights
                    if error > 0.5: 
                        error = 1 - error
                        p = -1 

                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
        
        
        # Caculate error
        EPS = 1e-10
        clf.alpha = 0.5 * np.log((1 - error)/(error + EPS))

        predictions = clf.predict(X)

        # Update weights 
        weights *= np.exp(-clf.alpha * y * predictions)
        weights /= np.sum(weights) # normalise

        self.clfs.append(clf) 


    def predict(self, X):
        '''Predict method'''

        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred 
