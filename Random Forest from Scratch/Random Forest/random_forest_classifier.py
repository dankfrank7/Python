from decicion_tree_classification import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class RandomForest():
    def __init__(self, n_trees=10, max_depth=2, min_samples_split=2, n_features=None):
        '''Constructor'''

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, Y):
        '''Train all the trees'''

        self.trees = []
        
        # Loop through every tree
        for i in range(self.n_trees):

            # Generate the tree
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            
            # Generate a randomised bootstrap dataset
            x_sample, y_sample, feature_idxs = self.bootstrap_samples(X, Y)

            # Fit the tree based on the dataset   
            tree.fit(x_sample, y_sample, feature_idxs)

            # Save the tree
            self.trees.append(tree)

            print('Tree ',i + 1,'/',self.n_trees,' Build Complete')
            print(feature_idxs)

    def bootstrap_samples(self, X, Y, feature_method='sqrt'):
        '''Generate bootstrap data '''
        
        # Randomly generate list of array of idexs 
        n_samples, n_features = X.shape

        # Sample bootstrap
        idxs = np.random.choice(n_samples, n_samples, replace=True) 

        # Feature bootstap 
        if feature_method =='sqrt':
            n_features_sample = int(np.floor(np.sqrt(n_features)))
        else: 
            n_features_sample = feature_method

        feature_idxs = np.sort(np.random.choice(n_features, n_features_sample, replace=False))
               
        return X[idxs], Y[idxs], feature_idxs

    def predict(self, X):
        '''Predict Y based on input X and method'''

        # Generate predictions for each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)

        most_common_labels = []
        percent_votes = []        

        for pred in tree_preds: 
            most_common, vote_percent = self.most_common_label(pred)
            most_common_labels.append(most_common)
            percent_votes.append(vote_percent)

        return most_common_labels, percent_votes
    
    def most_common_label(self, Y): 
        '''Calculate most common Y in predictions vec'''

        counter = Counter(Y)
        most_common, num_votes = counter.most_common(1)[0]

        percent_vote = num_votes / Y.shape[0]

        return most_common, percent_vote

    def plot_forest(self, forest=None, depth=0, pos_x=0, level_gap=20, ax=None):
        '''
        Plot the forst of trees
        
        Updated to plot in subplots
        '''

        if forest is None:
            forest = self.trees    

        # Set up plotting geometry
        n_plots = len(forest)
        n_rows = int(n_plots**0.5)
        n_cols = n_plots // n_rows if n_plots % n_rows == 0 else n_plots // n_rows + 1
        
        # Create figure and subplots 
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,8))
        axs = axs.flatten() # Flatten in case we have a grid

        # Loop through every tree in the forest
        for i, tree in enumerate(forest): 

            ax = axs[i]
            tree.plot_decision_tree(tree.root, depth, pos_x, level_gap, ax)
            ax.set_title('Tree {}'.format(i+1))
            ax.axis('off')

        # Hide any unused subplots
        for i in range(n_plots, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()