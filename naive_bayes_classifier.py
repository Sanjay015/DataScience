"""Naive Bayes Classifier."""
# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math


def load_csv(filename):
    """Read CSV."""
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def split_dataset(dataset, splitratio):
    """Split Data Set."""
    trainsize = int(len(dataset) * splitratio)
    trainset = []
    copy = list(dataset)
    while len(trainset) < trainsize:
        index = random.randrange(len(copy))
        trainset.append(copy.pop(index))
    return [trainset, copy]


def separate_by_class(dataset):
    """Class separation."""
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    """Mean calc."""
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    """Standard Deviation."""
    avg = mean(numbers)
    variance = sum(
        [pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(dataset):
    """Summarize data."""
    summaries = [(
        mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    """Summary by class."""
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries


def calculate_probability(x, mean, stdev):
    """Probability Calculations."""
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    """Class probability calc."""
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_probability(x, mean, stdev)
    return probabilities


def predict(summaries, input_vector):
    """Predict data."""
    probabilities = calculate_class_probabilities(summaries, input_vector)
    bestLabel, best_prob = None, -1
    for class_value, probability in probabilities.iteritems():
        if bestLabel is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    """Get predictions."""
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    """Model Accuracy."""
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    """Main Function."""
    filename = 'data/pima-indians-diabetes.csv'
    split_ratio = 0.67
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train={1} and test={2} rows').format(
        len(dataset), len(training_set), len(test_set))
    # prepare model
    summaries = summarize_by_class(training_set)
    # test model
    predictions = get_predictions(summaries, test_set)
    # print predictions
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%').format(accuracy)

main()
