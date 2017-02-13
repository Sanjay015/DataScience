"""Function to calculate GINI Score.

The Gini index is the name of the cost function used to evaluate splits
in the dataset.
A split in the dataset involves one input attribute and one value for that
attribute. It can be used to divide training patterns into two groups of rows.

A Gini score gives an idea of how good a split is by how mixed the classes are
in the two groups created by the split. A perfect separation results in a Gini
score of 0, whereas the worst case split that results in 50/50 classes in each
group results in a Gini score of 1.0 (for a 2 class problem).
Calculating Gini is best demonstrated with an example.

We have two groups of data with 2 rows in each group.
The rows in the first group all belong to class 0 and the rows in the second
group belong to class 1, so it's a perfect split.

We first need to calculate the proportion of classes in each group.
"""
# CART on the Bank Note dataset
# import pandas as pd
from csv import reader
from random import seed
from random import randrange


def load_csv(filename):
    """Load CSV file."""
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    # dataset = pd.read_csv(filename).as_matrix()
    return dataset


def str_column_to_float(dataset, column):
    """Convert string column to float."""
    for row in dataset:
        row[column] = float(row[column].strip())


def cross_validation_split(dataset, n_folds):
    """Split a dataset into k folds."""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage."""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """Evaluate an algorithm using a cross validation split."""
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def test_split(index, value, dataset):
    """Split a dataset based on an attribute and an attribute value."""
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, class_values):
    """Calculate the Gini index for a split dataset."""
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            grp = [row[-1] for row in group]
            proportion = grp.count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset):
    """Select the best split point for a dataset."""
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score = index, row[index], gini
                b_groups = groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    """Create a terminal node value."""
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    """Create child splits for a node or make terminal."""
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    """Build a decision tree."""
    root = get_split(dataset)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    """Make a prediction with a decision tree."""
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    """Classification and Regression Tree Algorithm."""
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data/data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth,
                            min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
