#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/01, homeway'

import sys, os, numpy as np
sys.path.append('/usr/local/lib/python2.7/site-packages')

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from config import Config as conf
import cv2

class Pattern(object):
    def __init__(self, conf = conf, clf = KNeighborsClassifier(), debug = False):
        self.clf = clf
        self.conf = conf
        self.debug = debug
        self.base = os.path.dirname(os.path.realpath(__file__))
        self.vote_db = {}
        self.letter_db = {}
        self.writer_db = {}
        self.total = self.right = 0

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def read(self, path):
        img = cv2.imread(path)
        if not img is None:
            img_data = []
            for x in range(len(img)):
                for y in range(len(img[x])):
                    img_data.append(img[x][y][0])
            return img_data
        return None

    def get_train_pair(self, letter):
        X_list = []
        y_list = []
        for i, p in enumerate(os.listdir(self.conf.train_path)):
            tmp_path = '{:s}/{:s}'.format(self.conf.train_path, p)
            if not os.path.isdir(tmp_path): continue
            for j, q in enumerate(os.listdir(tmp_path)):
                if q[0] == letter:
                    img_data = self.read('{:s}/{:s}'.format(tmp_path, q))
                    if not img_data is None:
                        X_list.append(img_data)
                        y_list.append(p)
        return np.array(X_list), y_list

    def init_voter(self):
        voter = {}
        writers = os.listdir(self.conf.train_path)
        for i, writer in enumerate(writers):
            if not os.path.isdir('{:s}/{:s}'.format(self.conf.train_path, writer)): continue
            voter[writer] = 0
        return voter

    def evaluate(self, predict, writer, letter):
        self.total += 1
        self.letter_db[letter]['total'] += 1
        if predict == writer:
            self.right += 1
            self.letter_db[letter]['right'] += 1

        self.letter_db[letter]['writer'].append(predict)
        self.letter_db[letter]['writer'] = list(set(self.letter_db[letter]['writer']))


    def run(self):
        self.letter_db = {}
        writers = os.listdir(self.conf.test_path)
        for i, writer in enumerate(writers):
            X = []
            y = []
            voter = self.init_voter()

            test_writer_path = '{:s}/{:s}'.format(self.conf.test_path, writer)
            if not os.path.isdir(test_writer_path): continue

            test_file_list = os.listdir(test_writer_path)

            for j, img_name in enumerate(test_file_list):
                img_data = self.read('{:s}/{:s}'.format(test_writer_path, img_name))
                if not img_data is None:
                    X.append(img_data)
                    y.append(writer)

                    # save letter data to self.letter_db
                    letter = img_name[0]
                    if letter not in self.letter_db: self.letter_db[letter] = {'right': 0, 'total': 0, 'writer': []}

                    # train && predict
                    X_list, y_list = self.get_train_pair(letter)
                    self.fit(X_list, y_list)
                    predict = self.predict(img_data)

                    # voter for winer of letter predict
                    voter[predict[0]] += 1
                    self.evaluate(predict[0], writer, letter)

            # writer evaluate
            self.vote_db[writer] = voter

    def show(self):
        keys = []
        values = []
        for (k, v) in self.letter_db.iteritems():
            total = v['total']
            right = v['right']
            keys.append(k)
            values.append(100 * float(right / float(total)))

        groups = len(self.letter_db)
        index = np.arange(groups)
        width = 0.5
        opacity = 0.4

        plt.bar(index, values, linewidth = width, alpha = opacity, color = 'b', label = 'right rate')

        plt.xlabel('letter')
        plt.ylabel('predict rgith rate (%)')
        plt.title('Writer identify: letter right rate')
        plt.xticks(index + width, keys)
        plt.ylim(0, 100)
        plt.legend()
        plt.show()

    def log(self):
        print('=====================================================>')
        global_right = global_total = 0
        for (k, v) in self.letter_db.iteritems():
            total = v['total']
            right = v['right']
            global_right += right
            global_total += total
            print('=> letter = {:s}, total = {:d}, right = {:d}, right_rate = {:.2f}%'.format(k, total, right, 100 * float(right / float(total))))
        print('=> letter predict: total = {:d}, right = {:d}, right_rate = {:.2f}%'.format(global_total, global_right, 100 * float(global_right / float(global_total))))

        print('\n\n=====================================================>')
        right = 0
        total = len(self.vote_db)
        for (k, v) in self.vote_db.iteritems():
            winer = (sorted(v.iteritems(), key = lambda d: d[1], reverse=True))[0][0]
            if winer == k: right += 1
            print('=> writer = {:s}, predict = {:s}'.format(k, v))
        print('=> writer predict: total = {:d}, right = {:d}, right_rate = {:.2f}%'.format(total, right, 100 * float(right / float(total))))

if __name__ == '__main__':
    from generater import Generate
    G = Generate(split = 0.3)
    # clear test file and generate test file
    G.recover().build()

    o = Pattern(debug = True)
    o.run()
    o.log()
    o.show()

    # remove test file
    G.recover()