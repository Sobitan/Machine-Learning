#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/06/20, grasses'

from knn import KNN
import csv, os

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/knn.csv'
    with open(dir_path, 'rb') as cf:
        lines = csv.reader(cf, quotechar=',')

        '''
        training dataset
        '''
        dataset = list(lines)

        '''
        set k = 3
        '''
        nn = KNN(dataset, k = 3, debug = True)

        '''
        start predict by some function
        '''
        print('===============test instance()=================')
        nn.test([5.7,2.8,4.1,1.3], 'Iris-versicolor')
        nn.test([7.1,3.0,5.9,2.1], 'Iris-versicolor')
        print('\n===============accuracy()=================')
        nn.accuracy()

        print('\n===============predict()=================')
        nn.predict([7.1,3.0,5.9,2.1])
        print('\n===============mult_predict()=================')
        nn.mult_predict([[4.6,3.4,1.4,1.3], [7.1,3.0,5.9,1.1], [2.3,3.3,4.5,6.7]])

        '''
        train from dataset && (0 ~ split) for test, others for test.
        split ∈ (0, 1)
        '''
        print('\n===============train_predict()=================')
        nn.train_predict(0.1)