#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/09, grasses'

import csv, random, math

class Bayesian(object):
    def __init__(self):
        self.dataset = []
        self.trainset = []
        self.testset = []
        self.summaries = {}

    @staticmethod
    def mean(numbers):
        return sum(numbers) / float(len(numbers))

    @staticmethod
    def variance(numbers):
        if len(numbers) - 1 == 0:
            raise ValueError('trainset not enough abundant! Each tag y has at least 2 data.')

        avg = Bayesian.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    '''
    suppose p(xi|c) ~ N(u, pow(σ, 2)), normal distribution
    :wikipedia  https://en.wikipedia.org/wiki/Normal_distribution
    :return p(xi|c) float
    '''
    @staticmethod
    def normal_distribution(x, c, u, sigma):
        exponent = math.exp(-(math.pow(x - u, 2) / (2 * math.pow(sigma, 2))))
        return (1 / (math.sqrt(2 * math.pi) * sigma)) * exponent

    '''
    suppose p(xi|c) ~ B(p), bernoulli distribution
    :wikipedia  https://en.wikipedia.org/wiki/Bernoulli_distribution
    :return p(xi|c) float
    '''
    @staticmethod
    def bernoulli_distribution(x, c, u, sigma):
        return pow(x, c) * pow(1 - x, 1 - c)

    '''
    load dataset, save in self.dataset, return self
    '''
    def load_db(self, path = r''):
        try:
            cf = open(path, 'rb')
        except IOError as e:
            print(e)
            return
        self.dataset = list(csv.reader(cf))
        for i in range(len(self.dataset)):
            self.dataset[i] = [float(x) for x in self.dataset[i]]
        return self

    '''
    split dataset into trainset && testset, return self
    :param split  float     split∈(0.0, 1.0), where split is the training set rate
    '''
    def split(self, split = 0.7):
        for i in range(len(self.dataset)):
            if random.random() < split:
                self.trainset.append(self.dataset[i])
            else:
                self.testset.append(self.dataset[i])
        return self

    '''
    separate key && value from list to dict, [[key1, key2, value1/value2], [key1, key2, value1/value2] ...] => {'value1': [...], 'value2': [...]}
    :param dataset  list    train dataset list
    :return separate dict   {'value1': [...], 'value2': [...]}
    '''
    def separate(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    '''
    summarize class
    :param instance list    [key1, key2, key3...]
    :return summaries list  [(mean, variance), (mean, variance)...], (mean, variance) for each list
    '''
    def summarize_instance(self, instances):
        summaries = [(self.mean(attribute), self.variance(attribute)) for attribute in zip(*instances)]
        del summaries[-1]
        return summaries

    '''
    summarize class, caculate each line mean and variance. return for example:
        summaries = {0: [(3.2040520984081042, 2.943976570806183)], 1: [(4.7466307277628035, 3.735940595196482), (140.63342318059298, 32.29325699645907)]}
    :return summarize   dict    {'value1': [(mean, variance), (mean, variance)...], 'value2': [...]}
    '''
    def summarize_class(self):
        separated = self.separate(self.trainset)
        self.summaries = {}
        for class_value, instances in separated.iteritems():
            self.summaries[class_value] = self.summarize_instance(instances)
        return self

    '''
    multiplied probabilities, return for example:
        probabilities = {0.0: 3.5571663857026466e-12, 1.0: 3.183595299286384e-13}
    :return probabilities   dict    {value1: probability1, value2: probability2}
    '''
    def calculate_class_probabilities(self, vector):
        probabilities = {}
        for class_value, class_summaries in self.summaries.iteritems():
            probabilities[class_value] = 1
            # multiplied probabilities => P(x) = π{p(xi|c), (i = 0, 1, 2, 3...)}
            for i in range(len(class_summaries)):
                (mean, stdev) = class_summaries[i]
                x = vector[i]
                probabilities[class_value] *= self.normal_distribution(x, class_value, mean, stdev)
            # after multiplied, predict(x) = max{Pj(x)| j = 0, 1, 2...len(self.summaries) - 1}
        return probabilities

    '''
    test for system
    :param  csv_path    string      csv dataset path
    :param  split       float       split∈(0.0, 1.0)
    :param  (trainset, testset, predictions)    please use with self.get_accuracy()
    '''
    def test(self, csv_path = r'', split = 0.7):
        self.load_db(csv_path)
        self.split(split)
        self.summarize_class()
        predictions = self.batch_predict(self.testset)
        return (self.trainset, self.testset, predictions)

    '''
    get accuracy only for test(), not for fit()/predict() accuracy
    :param      predictions list    predictions data [value1, value2, ...]
    :return     accuracy    float
    '''
    def get_accuracy(self, predictions):
        if len(predictions) == 0 or len(self.testset) == 0: return 0.0
        correct = 0
        for x in range(len(predictions)):
            if self.testset[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(predictions))) * 100.0

    '''
    fit() to set train dataset, len(X) = len(y)
    :param X    list    [[key1, key2, key3...], [...]]
    :param y    list    [value1, value2...]
    '''
    def fit(self, X, y):
        # reset dataset
        self.__init__()
        # fit new data
        for i in range(len(X)):
            X[i].append(y[i])
            self.trainset.append(X[i])
        self.summarize_class()
        return self

    '''
    predict single vector
    :param  x   list    [key1, key2...]
    '''
    def predict(self, x):
        probabilities = self.calculate_class_probabilities(x)
        best_label, best_prob = None, -1

        for class_value, probability in probabilities.iteritems():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    '''
    batch predict
    :param  X   list    [[key1, key2...], [...]]
    '''
    def batch_predict(self, X):
        if len(self.summaries) == 0: return []
        predictions = []
        for i in range(len(X)):
            predictions.append(self.predict(X[i]))
        return predictions