import numpy as np
import pandas as pd
from id3 import ID3DecisionTree
from decisiontree import DecisionTreeWithPruning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read from the data
def load_data(path):
    df = pd.read_csv(path, header=None)
    # read from data to get all features
    X = df.iloc[:, 2:].values
    # read from the data to get all labels where M is 1 and B is 0
    y = df.iloc[:, 1].values
    labelList = []
    for label in y:
        if label == 'M':
            labelList.append(1)
        elif label == 'B':
            labelList.append(0)
        else:
            continue
    labelList = np.array(labelList)
    return X, labelList

def main():
    X, y = load_data('data/wdbc.data')
    # Split the dataset into training, validation and testing sets
    # Where the first 70% for train, 10% for validation and last 20% for testing
    n_samples = len(X)
    trainSplit = int(0.7*n_samples)
    valSplit = int(0.8*n_samples)
    X_train, y_train = X[:trainSplit], y[:trainSplit]
    X_val, y_val = X[trainSplit:valSplit], y[trainSplit:valSplit]
    X_test, y_test = X[valSplit:], y[valSplit:]

    # Implementation on ID3 Decision Tree
    ID3tree = ID3DecisionTree(max_depth=10) 
    ID3tree.fit(X_train, y_train)

    y_valPred = ID3tree.predict(X_val)
    y_testPred = ID3tree.predict(X_test)
    print(f"ID3 Tree - Validation Accuracy: {accuracy_score(y_val, y_valPred):.4f}")
    print(f"ID3 Tree - Test Accuracy: {accuracy_score(y_test, y_testPred):.4f}")

    # Implementation on Decistion Tree
    decisionTree = DecisionTreeWithPruning(max_depth=10)
    decisionTree.fit(X_train, y_train)
    #y_valPred_DT = decisionTree.predict(X_val)
    #y_testPred_DT = decisionTree.predict(X_test)
    #print("True Validation Labels (y_val):", y_val[:10])  # Print first 10 labels
    #print("Predicted Validation Labels (y_valPred_DT):", y_valPred_DT[:10])  # Print first 10 predicted labels

    decisionTree.pruning(decisionTree.root, X_val, y_val)
    y_valPred_DTP = decisionTree.predict(X_val)
    y_testPred_DTP = decisionTree.predict(X_test)
    #print("Predicted Validation Labels (y_valPred_DTP):", y_valPred_DTP[:10])  # Print first 10 predicted labels
    print(f"Decision Tree after Pruning - Validation Accuracy: {accuracy_score(y_val, y_valPred_DTP):.4f}")
    print(f"Decision Tree after Pruning - Test Accuracy: {accuracy_score(y_test, y_testPred_DTP):.4f}")


if __name__ == "__main__":

    main()