import numpy as np 

class SVM(): 
    '''
    Support Vector Machine from scratch
    - Linear kernel function
    '''
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        '''Constructor'''
        
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None 
        self.b = None 

    def fit(self, X: np.ndarray, y: np.array): 
        '''Fit method'''

        n_samples, n_features = X.shape 

        # Split into classes
        y_ = np.where(y <= 0, -1, 1)

        # Init weights (better to randomly initialise)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters): 
            for idx, x_i in enumerate(X): 

                #self.visualize_svm(X, y)

                # Check whether each point is being classified correctly
                condition = y_[idx] * np.dot(x_i, self.w) - self.b >= 1 

                # If correct classificaiton
                if condition: 

                    # If correct classification, then the only w update is the regularisation
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    self.b -= 0 

                # If incorrect classification
                else: 

                    # If incorrect classification then update by dJ/dw & dJ/db
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X: np.ndarray): 
        '''Predict method'''
        
        return np.sign(np.dot(X, self.w) - self.b)
    
    def get_hyperplane_value(self, x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    def visualize_svm(self, X: np.ndarray, y: np.array):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = self.get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = self.get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = self.get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = self.get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = self.get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = self.get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show(block=False)
        input('Press enter to close figure...')
        plt.close()


# Test the code 
    
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=500, n_features=2, centers=2, cluster_std=1.25, random_state=41
    )
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)
    # predictions = clf.predict(X)

    print(clf.w, clf.b)
    clf.visualize_svm(X, y)

    #visualize_svm()