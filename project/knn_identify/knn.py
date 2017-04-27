#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/04/25, homeway'

from numpy import *
import operator, numpy as np
from os import listdir

class KNN_IMG(object):
    def __init__(self, test_dir = 'test_digits', train_dir = 'training_digits'):
        import os
        self.base = os.path.dirname(os.path.realpath(__file__))

        self.test_dir = self.base + '/' + test_dir
        self.train_dir = self.base + '/' + train_dir
        self.train_X = []
        self.train_Y = []

    def fit(self, train_dir = 'training_digits'):
        self.train_dir = self.base + '/' + train_dir
        training_file_list = listdir(self.train_dir)
        size = len(training_file_list)

        self.train_X = np.zeros((size, 1024))
        for i in range(size):
            name = training_file_list[i]
            fid = int((name.split('.')[0]).split('_')[0])
            self.train_Y.append(fid)
            self.train_X[i, :] = self.image2vector('{:s}/{:s}'.format(self.train_dir, name))

    def image2vector(self, name):
        vector = zeros((1, 1024))
        with open(name) as fd:
            for x in range(32):
                line = fd.readline()
                for y in range(32):
                    vector[0, 32 * x + y] = int(line[y])
        return vector

    def knn(self, test_X = [], k = 3):
        size = self.train_X.shape[0]

        # Euclidean algorithm
        diff = tile(test_X, (size, 1)) - self.train_X
        dist_pow2 = diff ** 2
        dist_sum = dist_pow2.sum(axis = 1)
        dist_sqrt = dist_sum ** 0.5
        dist = dist_sqrt.argsort()

        # vote for neighbors
        class_count = {}
        for i in range(k):
            vote_label = self.train_Y[dist[i]]
            class_count[vote_label] = class_count.get(vote_label, 0) + 1
        sorts = sorted(class_count.iteritems(), key = operator.itemgetter(1), reverse = True)
        return sorts[0][0]

    def test(self, test_dir = 'test_digits', k = 3):
        self.test_dir = self.base + '/' + test_dir
        test_file_list = listdir(self.test_dir)

        error_count = 0
        total_count = len(test_file_list)
        for i in range(total_count):
            name = test_file_list[i]
            fid = int((name.split('.')[0]).split('_')[0])
            vector = self.image2vector('{:s}/{:s}'.format(self.test_dir, name))
            result = self.knn(test_X = vector, k = k)
            if (result != fid):
                error_count += 1.0
            print('=> The knn predict = {:d}, real answer = {:d}'.format(result, fid))
        print('\n=> Error predict times = {:.2f}, error rate = {:.2f}%'.format(error_count, 100 * float(error_count / total_count)))

if __name__ == '__main__':
    knn = KNN_IMG()
    knn.fit()
    knn.test(k = 10)