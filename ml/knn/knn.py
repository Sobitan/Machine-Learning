#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import numpy as np, csv, operator, os

class KNN(object):
    def __init__(self, tlist, k = 5, debug = True):
        self.debug = debug
        self.k = k
        self.train_list = tlist
        self.test_list = []
        self.predictions = []
        self.right_count = 0
        self.total_count = 0
    '''
    function fit() set training && testing data
    @input  train_list  list    training list
    @input  test_list   list    testing list
    @return None
    '''
    def fit(self, train_list = [], test_list = []):
        self.train_list = train_list
        self.test_list = test_list

    '''
    reset class to default
    '''
    def reset(self):
        self.__init__([], k = self.k, debug = self.debug)

    '''
    test() function is to predict single instance
    @input  instance    list    testing list
    @return predict     string  predict value
    @return  knn algorithm predict result
    '''
    def test(self, instance = [], predict = '', type = 1):
        neighbors = self.get_neighbors(instance)
        result = self.get_min_dest(neighbors)
        if str(result) == str(predict):
            self.right_count += 1
        self.total_count += 1

        if type == 1:
            # append test instance to self.tets_list by default
            instance.append(predict)
            self.test_list.append(instance)

        if self.debug == True:
            print('test()-> knn predicted = ' + repr(result) + ', your given actual = ' + repr(predict))
        return result

    '''
    train_predict() function is to testing from self.train_list, you can load data by fit() or datalist
    @input  split       float   (0 ~ split) possibility for test, others for test, split ∈ (0, 1)
    @return datalist    list    predict data list
    @return None
    '''
    def train_predict(self, split = 0.5, datalist = None):
        if datalist != None:
            tmp = self.train_list
        else:
            tmp = self.train_list
        self.reset()

        for x in range(len(tmp)):
            if random.random() < split:
                self.test_list.append(tmp[x])
            else:
                self.train_list.append(tmp[x])

        for x in range(len(self.test_list)):
            y = self.test_list[x][len(self.test_list[x]) - 1]
            self.test_list[x].pop(-1)
            self.test(self.test_list[x], y, type = 2)

        # show accuracy after test
        self.accuracy()

    '''
    mult_predict() function call predict() one by one
    @input  xlist       list        [[], [], []]format
    @return predict     list        []format
    '''
    def mult_predict(self, xlist):
        result = []
        for x in range(len(xlist)):
            result.append(self.predict(xlist[x]))
        return result

    '''
        predict()
        @input  instance    list        []format
        @return predict     string      predict result
        '''
    def predict(self, instance = []):
        neighbors = self.get_neighbors(instance)
        result = self.get_min_dest(neighbors)
        self.predictions.append((instance, result))

        if self.debug == True:
            print('predict()-> instance = ' + str(instance) + ', knn predict = ' + result)
        return result

    def get_min_dest(self, neighbors = []):
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_vote = sorted(class_votes.iteritems(), key = operator.itemgetter(1), reverse=True)
        return sorted_vote[0][0]

    def get_neighbors(self, instance):
        distances = []
        neighbors = []
        for x in range(len(self.train_list)):
            distances.append((self.train_list[x], self.get_dest(instance, self.train_list[x])))
        distances.sort(key = operator.itemgetter(1))

        if self.k < len(distances[0]):
            self.k = len(distances[0])

        for i in range(self.k):
            neighbors.append(distances[i][0])
        return neighbors

    def get_dest(self, sour, dest):
        distance = 0
        for i in range(len(sour) - 1):
            distance += pow((float(sour[i]) - float(dest[i])), 2)
        return math.sqrt(distance)

    def accuracy(self):
        print('training size = {:d}, testing size = {:d}'.format(len(self.train_list), len(self.test_list)))
        print('accuracy()-> total instance = {:d}, right instance = {:d}, error instance = {:d}, right rate = {:.1f}%'.format(self.total_count, self.right_count, self.total_count - self.right_count, 100 * float(self.right_count)/float(self.total_count)))

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