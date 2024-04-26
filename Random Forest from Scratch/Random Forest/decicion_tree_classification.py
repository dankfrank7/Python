import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        '''constructor'''

        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # For leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):   
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def clear_tree(self):
        '''Clears tree for use in random forest'''

        self.root = None
    
    def build_tree(self, dataset, feature_idxs, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if self.is_deciscion_node(dataset, num_samples, curr_depth, feature_idxs):
            
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features, feature_idxs)
            
            # check if information gain is positive
            if best_split["info_gain"]>0:
                
                # recur left
                left_subtree  = self.build_tree(best_split["dataset_left"], feature_idxs, curr_depth+1)
                
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], feature_idxs, curr_depth+1)
                
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)

        # return leaf node
        return Node(value=leaf_value)
    
    def is_deciscion_node(self, dataset, num_samples, curr_depth, feature_idxs):
        '''Checks if node is a decicison node, otherwise it's a leaf node'''

        # If the number of samples is below min samples then it is a leaf node
        if num_samples < self.min_samples_split:
            return False
        
        # If the current depth is larger than maximum, then it is a leaf node
        if curr_depth > self.max_depth:
            return False
        
        # If the dataset is pure (only one Y) then it is leaf node
        if self.ispure(dataset):
            return False
        
        # If there is no way to split the dataset, (only one unique value of each feature index) then it is a leaf node
        return any(len(np.unique(dataset[:, i])) > 1 for i in feature_idxs)
        
    
    def ispure(self, dataset):
        '''Function checks if dataset is purely one classifications'''

        dataset_Y = dataset[:,-1]
        first_Y = dataset_Y[0]

        return all(item == first_Y for item in dataset_Y)
            
    def get_best_split(self, dataset, num_samples, num_features, feature_idxs):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in feature_idxs:
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
           
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    
                    # Extract Y datasets
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, mode="gini")

                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode=None):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", round(tree.info_gain,2))
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y, feature_idxs):
        ''' function to train the tree '''

        # Store the indexes of the current variables
        self.feature_idxs = feature_idxs
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset, feature_idxs)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        # First check if it a Leaf node, if it is return the value
        if tree.value!=None: 
            return tree.value

        # Otherwise the feature index from the decicsion node        
        feature_val = x[tree.feature_index]

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def plot_decision_tree(self, tree=None, depth=0, pos_x=0, level_gap=20, ax=None):
        """
        Recursively plot the decision tree.

        Args:
        - node: The current node to be plotted.
        - depth: The depth of the current node in the tree.
        - pos_x: The horizontal position of the node.
        - level_gap: Vertical gap between levels of the tree.
        - ax: Matplotlib Axes object.
        """

        if tree is None:
            tree = self.root

        plt.figure

        if ax is None:
            ax = plt.gca()
            ax.clear()
            ax.set_ylim(0, 100)

        # Base case: if the node is a leaf
        if tree.left is None and tree.right is None:
            ax.text(pos_x, 100 - depth * level_gap, f'Leaf\nValue: {tree.value}', 
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
            return

        # Decision node
        ax.text(pos_x, 100 - depth * level_gap, f'X[{tree.feature_index}] <= {tree.threshold:.2f}\nGain: {tree.info_gain:.2f}', 
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=2))

        # Positioning of the left and right children
        pos_x_left = pos_x - (10 - depth * 2)
        pos_x_right = pos_x + (10 - depth * 2)

        # Connect current node to left and right children
        ax.plot([pos_x, pos_x_left], [100 - depth * level_gap, 100 - (depth + 1) * level_gap], 'k-')
        ax.plot([pos_x, pos_x_right], [100 - depth * level_gap, 100 - (depth + 1) * level_gap], 'k-')

        # Recursively plot left and right children
        if tree.left is not None:
            self.plot_decision_tree(tree.left, depth + 1, pos_x_left, level_gap, ax)

        if tree.right is not None:
            self.plot_decision_tree(tree.right, depth + 1, pos_x_right, level_gap, ax)

        #if depth == 0:
        #    ax.axis('off')
        #    plt.show()