#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/01, homeway'

from config import Config as conf
import os, random, sys, time, numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

class Generate(object):
    def __init__(self, config = conf, split = 0.3, clf = KNeighborsClassifier(), auto_rebuild = False, debug = False):
        self.clf = clf
        self.conf = conf
        self.split = split
        self.debug = debug
        self.auto_rebuild = auto_rebuild
        self.init()

    def init(self):
        if not os.path.exists(self.conf.tmp_path): os.mkdir(self.conf.tmp_path)
        if not os.path.exists(self.conf.test_path): os.mkdir(self.conf.test_path)
        if not os.path.exists(self.conf.train_path): os.mkdir(self.conf.train_path)
        if not os.path.exists(self.conf.source_path): os.mkdir(self.conf.source_path)
        return self

    def rander(self):
        writers = os.listdir(conf.train_path)
        for writer in writers:
            test_writer_path = '{:s}/{:s}'.format(conf.test_path, writer)
            train_writer_path = '{:s}/{:s}'.format(conf.train_path, writer)

            if not os.path.isdir(train_writer_path): continue

            # make sure path: {test_path}/{writer} exist
            if not os.path.exists(test_writer_path):
                os.mkdir(test_writer_path)

            # move train file as test file
            files = os.listdir('{:s}/{:s}'.format(conf.train_path, writer))
            for file in files:
                # (0, split) to move train file => test file
                if random.random() < self.split:
                    os.rename('{:s}/{:s}'.format(train_writer_path, file), '{:s}/{:s}'.format(test_writer_path, file))
        return self

    def recover(self):
        writers = os.listdir(conf.test_path)
        for writer in writers:
            test_writer_path = '{:s}/{:s}'.format(conf.test_path, writer)
            train_writer_path = '{:s}/{:s}'.format(conf.train_path, writer)

            if not os.path.isdir(train_writer_path): continue

            # make sure path: {train_path}/{writer} exist
            if not os.path.exists(train_writer_path):
                os.mkdir(train_writer_path)

            # move test file to train file
            files = os.listdir('{:s}/{:s}'.format(conf.test_path, writer))
            for file in files:
                os.rename('{:s}/{:s}'.format(test_writer_path, file), '{:s}/{:s}'.format(train_writer_path, file))
        return self

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

    def build_train(self, ignore = ''):
        X_list = []
        y_list = []
        writers = os.listdir(conf.train_path)
        for writer in writers:
            # ignore youself
            if writer == ignore: continue

            train_writer_path = '{:s}/{:s}'.format(conf.train_path, writer)
            if not os.path.isdir(train_writer_path): continue

            files = os.listdir(train_writer_path)
            for file in files:
                img_data = self.read('{:s}/{:s}'.format(train_writer_path, file))
                if img_data is None: continue

                X_list.append(img_data)
                y_list.append(file[0])
        X = np.array(X_list)
        y = np.array(y_list)
        return self.fit(X, y)

    def build_identify(self, writer):
        X_list = []
        dict_map = {}

        train_writer_path = '{:s}/{:s}'.format(conf.train_path, writer)
        files = os.listdir(train_writer_path)
        for file in files:
            tmp_path = '{:s}/{:s}'.format(train_writer_path, file)
            img_data = self.read(tmp_path)
            if img_data is None: continue
            X_list.append(img_data)

            letter = self.predict(np.array(img_data))[0]
            if letter not in dict_map:
                dict_map[letter] = 0
            else:
                dict_map[letter] += 1
            print('=> rename: {:s} => {:s}_{:d}.jpg'.format(file, letter, dict_map[letter]))
            os.rename(tmp_path, '{:s}/{:s}_{:d}.jpg'.format(train_writer_path, letter, dict_map[letter]))
        return self

    def rebuild(self, writer):
        self.build_train(writer)
        self.build_identify(writer)

    def cutting(self, writer, fname):
        fpath = '{:s}/{:s}/{:s}'.format(self.conf.source_path, writer, fname)
        img = cv2.imread(fpath)
        if img is None: return
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, im_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        if self.debug:
            print('=> file path = {:s}'.format(fpath))

        for i, rect in enumerate(rects):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            cv2.putText(img, '({:d},{:d})'.format(rect[0], rect[1]), (rect[0], rect[1]), font, 0.8, (0, 255, 0), 2)
            cv2.putText(img, '({:d},{:d})'.format(rect[0] + rect[2], rect[1] + rect[3]), (rect[0] + rect[2], rect[1] + rect[3]), font, 0.8, (0, 255, 0), 2)

            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

            if pt1 < 0 or pt2 < 0: continue

            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]

            print("i = {:d} leng = {:.0f} pt1 = {:d} pt2 = {:d} rect[0] = {:d} rect[1] = {:d} rect[2] = {:d} rect[3] = {:d}".format(i, leng, pt1, pt2, rect[0], rect[1], rect[2], rect[3]))

            from matplotlib import pyplot
            import matplotlib as mpl
            fig = pyplot.figure()
            ax = fig.add_subplot(1, 1, 1)
            imgplot = ax.imshow(roi, cmap=mpl.cm.Greys)
            imgplot.set_interpolation('nearest')
            ax.xaxis.set_ticks_position('top')
            ax.yaxis.set_ticks_position('left')
            #pyplot.show()

            roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            #roi_hog_fd = hog(roi, orientations = 9, pixels_per_cell = (14, 14), cells_per_block = (1, 1), visualise = False)
            cv2.imwrite('{:s}/{:s}/img.{:d}.{:.2f}.jpg'.format(self.conf.train_path, writer, i, time.time()), roi)
        cv2.imwrite('{:s}/img.{:d}.jpg'.format(self.conf.tmp_path, i), img)

    def build(self):
        dict_list = {}
        writers = os.listdir(conf.source_path)
        for writer in writers:
            train_writer_path = '{:s}/{:s}'.format(conf.train_path, writer)
            source_writer_path = '{:s}/{:s}'.format(conf.source_path, writer)

            if not os.path.isdir(source_writer_path): continue

            # make sure path: {train_path}/{writer} exist
            if not os.path.exists(train_writer_path):
                os.mkdir(train_writer_path)
                print('=> mkdir: {:s} '.format(train_writer_path))
            else:
                print('=> {:s} writer exist.'.format(writer))
                continue

            # build train file
            files = os.listdir(source_writer_path)
            for file in files:
                self.cutting(writer, file)

            # auto rename file to {y_label}_{index}.jpg, using knn algorithm
            if self.auto_rebuild:
                self.rebuild(writer)

        return self

if __name__ == '__main__':
    G = Generate(split = 0.2, auto_rebuild = True)
    G.recover().rander()
    # G.build() build new train files from source folder