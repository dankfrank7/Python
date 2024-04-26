import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.001, n_iters=1000):
        '''Constructor'''

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.array): 
        '''Optimise weights and bias using gradient descent'''

        # Initialise paramaters
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iters):
            
            # Calculate predictions
            y_predicted = self._make_predictions(X)

            # Calculate gradient of cost function 
            dw, db = self._calculate_gradients(X, y_predicted, y)  

            # Update weights and bias scaled by learning rate
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray):
        '''Calculate predictions from the model'''
        
        # Get predictions
        y_predicted = self._make_predictions(X)
    
        # Classify 
        y_predicted_cls = [i if i > .5 else 0 for i in y_predicted]

        return y_predicted_cls

    def _calculate_gradients(self, X, y_predicted, y_true): 
        '''Calcualtes the gradient of the cross-entropy cost functions'''
        
        n_samples = X.shape[0]

        dw = (1 / n_samples) * np.dot(X.T, y_predicted-y_true)
        db = (1 / n_samples) * np.sum(y_predicted-y_true)
        return dw, db

    def _make_predictions(self, X: np.ndarray) -> np.array: 
        '''Calculate predicted y'''
        
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, x: np.array):
        ''' Calculates sigmoid function of x'''
        
        return 1 / (1 + np.exp(-x)) 
