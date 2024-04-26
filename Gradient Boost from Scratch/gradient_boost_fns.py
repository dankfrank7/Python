from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeRegressor 
import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.metrics import r2_score


def normalise(df: pd.DataFrame) ->pd.DataFrame: 
    '''Normalise columns (uniform)'''

    df_norm = (df - df.min()) / (df.max() - df.min())

    return df_norm

def feature_hists(X: pd.DataFrame):
    '''Plot histograms for all featurs in X'''

    # Generage figures and axes
    num_features = X.shape[1]
    fig, axes = plt.subplots(1, num_features, figsize=(num_features*2,6), sharey=True)

    # Colormap
    cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
    colors = cmap(np.linspace(0, 1, num_features))
    
    # Plot in each axes
    for i, ax in enumerate(axes): 
        sns.histplot(X, ax=ax, x=X.iloc[:,i], kde=True, color=colors[i])

    plt.show(block=False)
    return fig, axes

class GradientBoostRegressor():   
    '''
    From Scratch: Gradient Boost Regressor v1.0

    Perforance: currently performs on par with sklearn.GradientBoostingRegressor for low max_depth values. For max_depth > 2, performance deteriorates rapidly

    To Do: 
    - figure out why tree size above 2 negatively effects performance
    - improve runtime performance
    '''
    def __init__(self, num_trees=10, max_depth=3, min_samples_split=5, learning_rate=0.1): 
        '''Constructor'''

        self.max_depth = max_depth
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.trees = []
        self.train_data = {'Predictions':pd.DataFrame(), 'Residuals':pd.DataFrame()}
        self.learning_rate = learning_rate

    def fit(self, X_train: pd.DataFrame, Y_train: pd.Series):
        '''Gradient descent boosting model (tree based)'''

        # First guess is the mean value of all the target variables
        self.root = np.mean(Y_train)

        # First iteration is residual between Y_train and self.root
        target = Y_train - self.root

        for i in range(self.num_trees):

            # 1) Fit model (sklearn.DecisionTreeRegressor)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split) 
            model = tree.fit(X_train, target)

            # 2) Predict values 
            predicted_y = pd.Series(model.predict(X_train), index=X_train.index)

            # 3) Calculate difference
            errors = target - self.learning_rate * predicted_y

            # 4) Save values
            itername = "Iteration " + str(i)
            self.trees.append(model)
            #self.train_data['Residuals'][itername] = errors
            #self.train_data['Predictions'][itername] = predicted_y # Causing fragmented dataframe warnings, not sure how to fix
            
            # 5) Update target
            target = errors

        return self 
    
    def predict(self, X_test: pd.DataFrame): 
        '''Use models to predict Y_test from X_test'''

        # First prediction is the mean
        predictions = []

        # Calculate predictions for every model
        for model in self.trees:
            predictions.append(self.learning_rate * model.predict(X_test))

        # Sum predictions of every model
        return pd.Series(np.sum(predictions, axis=0), index=X_test.index) + self.root
    
    def plot_residuals(self, log_scale=False):
        '''Plots residuals for this dataset'''

        self.check_if_trained()

        # Create a figure and axis for the plot
        plt.figure(figsize=(10, 6))

        residuals = []

        # Loop through each iteration and plot the residuals
        for itername in self.train_data['Residuals'].columns:
            residuals.append(np.sum(abs(self.train_data['Residuals'][itername])))
        
        plt.plot(range(self.num_trees), residuals)

        # Adding labels and title for clarity
        plt.xticks(np.arange(0,self.num_trees,5))
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.grid(which='major',color='k')
        plt.grid(which='minor',color='gray',alpha=.4)
        if log_scale:
            plt.yscale('log')

        plt.show(block=False)

    def check_if_trained(self):
        '''Checks if model has been trained'''

        if len(self.trees)==0:
            raise ValueError('Tree has not been trained')
        
    def tune(self, X_train, Y_train, X_test, Y_test, params=None) -> np.array: 
        '''Tune hyperparameters to find maximum'''

        if params is None:
            params = {
                'num_trees':[50, 100, 200], 
                'max_depth':[2, 4, 10, 50]
            }

        accuracy = np.zeros((len(params['num_trees']), len(params['max_depth'])))

        for i, num_trees in enumerate(params['num_trees']):
            for j, max_depth in enumerate(params['max_depth']):

                # Update features 
                self.num_trees = num_trees
                self.max_depth = max_depth

                # Fit model
                self.trees = []
                self.fit(X_train, Y_train)

                # Make predictions
                Y_predict = self.predict(X_test)

                # Calculate accuracy and save to the accuracy array
                accuracy[i,j] = r2_score(Y_test, Y_predict)

        # Find the maximum 
        i_max, j_max = self.find_max(accuracy)

        # Save the best results in the model
        self.num_trees = params['num_trees'][i_max]
        self.max_depth = params['max_depth'][j_max]

        return accuracy

    def find_max(self, matrix):
        '''Finds the i,j index of the maximum value in a 2D np.array'''

        flat_indices = np.argmax(matrix)
        i_max, j_max = np.unravel_index(flat_indices, matrix.shape)
        return i_max, j_max