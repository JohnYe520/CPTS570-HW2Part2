import numpy as np
import pandas as pd

# calculate the entropy
def entropy(y):
    labels, counts = np.unique(y, return_counts=True)
    totalLabel = len(y)
    entro = -np.sum((counts/totalLabel)*np.log2(counts/totalLabel))
    return entro

# Implementation on calculating the information gain
def info_gain(X, y, threshold):
    # H(S)
    totalEntropy = entropy(y)
    # initialize the left and right nodes
    left_node = []
    right_node = []
    for i in range(len(X)):
        if X[i]<= threshold:
            left_node.append(y[i])
        else:
            right_node.append(y[i])
    # No info gain if the node is empty
    if len(left_node) == 0 or len(right_node) == 0:
        return 0
    
    #calculate the conditional entropy (H(S|A)
    n = len(y)
    n_left = len(left_node)
    n_right = len(right_node)

    left_entropy = entropy(left_node)
    right_entropy = entropy(right_node)
    condition_entropy = (n_left/n) * (left_entropy) + (n_right/n) * (right_entropy)

    infoGain = totalEntropy - condition_entropy
    return infoGain

# Implementation on finding the best split 
def split(X,y):
    best_gain = -1
    best_feature = None
    best_threshold = None
    n_features = X.shape[1]

    for feature in range(n_features):
        X_col = X[:, feature]
        # Sort all candidate value for feature f from trainning data
        sortedValue = np.sort(np.unique(X_col))
        thresholds = []
        # The candidate thresholds are chosen as f_i + (f_(i+1) - f_i) / 2
        for i in range(len(sortedValue) - 1):
            threshold = (sortedValue[i] + (sortedValue[i+1] - sortedValue[i])/2)
            thresholds.append(threshold)

        for threshold in thresholds:
            gain = info_gain(X_col, y, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_node(self):
        if self.value != None:
            pass
            #print("This is a leaf node")
        return self.value
    
# Implementation for ID3 Decision Tree
class ID3DecisionTree:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.tree(X,y)

    def tree(self, X, y, depth = 0):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_labels = len(np.unique(y))

        # Check for whether the max depth is reached, or current node is pure node, or no more samples left.
        if (depth >= self.max_depth or n_labels== 1 or n_samples == 0):
            # assign the most commen label to the leaf node is the cases above
            leaf_value = self.commenLabel(y)
            return Node(value = leaf_value)

        # find the best feature and threshold to split, it no such feature, create a new leaf based on the most common node
        feature, threshold = split(X,y)
        if feature == None:
            leaf_value = self.commenLabel(y)
            return Node(value = leaf_value)

        # split the dataset besed on the feature and threshold
        left_nodes = []
        right_nodes = []
        for i in range(len(X)):
            if X[i, feature]<= threshold:
                left_nodes.append(i)
            else:
                right_nodes.append(i)

        left_nodes = np.array(left_nodes)
        right_nodes = np.array(right_nodes)

        # build the tree recursively
        left_tree = self.tree(X[left_nodes], y[left_nodes], depth+1)
        right_tree = self.tree(X[right_nodes], y[right_nodes], depth+1)

        return Node(feature, threshold, left_tree, right_tree)
    
    # Implementation on finding the most common label
    def commenLabel(self, y):
        labels, counts = np.unique(y, return_counts=True)
        commenlabel = np.argmax(counts)
        return labels[commenlabel]
    
    # Implementation on predictition
    def predict(self, X):
        predList = []
        for sample in X:
            pred = self.search(sample, self.root)
            predList.append(pred)
        
        predList = np.array(predList)

        return predList

    # Implementation on searching the tree to make a prediction
    def search(self, x, node):
        if node.leaf_node():
            return node.value
        
        if node.feature is None or node.threshold is None:
            return node.value
        # recursively search on left or right branch based on the feature and threshold 
        if x[node.feature] <= node.threshold:
            return self.search(x,node.left)
        else:
            return self.search(x,node.right)