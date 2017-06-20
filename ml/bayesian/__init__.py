#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/09, grasses'

import os
from bayes import Bayesian

if __name__ == '__main__':
    '''
    test datase from uci: https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
    '''

    B = Bayesian()
    csv_path = os.path.dirname(os.path.realpath(__file__)) + '/data/pima-indians-diabetes.data'
    # test() and get_accuracy()
    (trainset, testset, predictions) = B.test(csv_path=csv_path, split=0.7)
    accuracy = B.get_accuracy(predictions)
    print('=> accuracy: {:.2f}%\n').format(accuracy)
    exit(1)
    # fit(X, y)
    X = [[0,1,2,3,4], [1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]]
    y = [0,0,1,1]
    B.fit(X, y)

    # predict(x) and predict(X)
    x = [[4,5,6,7,8], [5,6,7,8,9]]
    predict = B.predict(x[0])
    predicts = B.batch_predict(x)
    print('=> predict({0}) = {1}'.format(X[0], predict))
    print('=> batch_predict({0}) = {1}'.format(X, predicts))