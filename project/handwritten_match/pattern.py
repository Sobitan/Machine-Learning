#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/04/27, homeway'

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2, os
from sklearn.svm import SVC
import numpy as np

class IMG(object):
    def __init__(self, input = 'source', output = 'train', debug = False):
        self.debug = debug
        self.base = os.path.dirname(os.path.realpath(__file__))
        self.input = '{:s}/{:s}'.format(self.base, input)
        self.output = '{:s}/{:s}'.format(self.base, output)

    def build(self, index, fname):
        if fname == '.DS_Store':
            return

        fpath = '{:s}/{:s}'.format(self.input, fname)
        print(fpath)
        img = cv2.imread(fpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, im_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)
        ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        opath = '{:s}/{:d}'.format(self.output, index)
        if not os.path.exists(opath):
            os.mkdir(opath)
            if self.debug:
                print('=> build path = {:s}'.format(opath))
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

            if pt1 < 0 or pt2 < 0:
                continue

            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]

            print("i = {:d} leng = {:f} pt1 = {:d} pt2 = {:d} rect[0] = {:d} rect[1] = {:d} rect[2] = {:d} rect[3] = {:d}".format(i, leng, pt1, pt2, rect[0], rect[1], rect[2], rect[3]))

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
            cv2.imwrite('{:s}/img{:d}.jpg'.format(opath, i), roi)
        cv2.imwrite('{:s}/img.jpg'.format(opath), img)

    def clean(self):
        olist = os.listdir(self.output)
        for i, o in enumerate(olist):
            opath = '{:s}/{:s}'.format(self.output, o)
            for j, x in enumerate(os.listdir(opath)):
                os.remove('{:s}/{:s}'.format(opath, x))
            os.rmdir('{:s}/{:s}'.format(self.output, o))
            if self.debug:
                print('=> remove {:s}/{:s}'.format(self.output, o))

    def dfs(self, base = 'train', is_root = True):
        olist = os.listdir(self.output)
        for i, o in enumerate(olist):
            tmp = '{:s}/{:s}'.format(base, o)
            self.dfs(tmp, False)
            if is_root == False:
                os.rmdir(base)

    def run(self):
        img_list = os.listdir(self.input)
        for i, o in enumerate(img_list):
            self.build(index = i, fname = o)

class svm(object):
    def __init__(self, train = 'train', test = 'test', debug = False):
        self.debug = debug
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = SVC()
        self.base = os.path.dirname(os.path.realpath(__file__))
        self.train_path = '{:s}/{:s}'.format(self.base, train)
        self.test_path = '{:s}/{:s}'.format(self.base, test)

        self.right = 0
        self.total = 0

        self.reality_list = []
        self.predict_list = []

    def train(self, X, y):
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

    def get_train_list(self, letter):
        X_list = []
        y_list = []
        for i, p in enumerate(os.listdir(self.train_path)):
            tmp_path = '{:s}/{:s}'.format(self.train_path, p)
            if not os.path.isdir(tmp_path): continue
            for j, q in enumerate(os.listdir(tmp_path)):
                if q[0] == letter:
                    img_data = self.read('{:s}/{:s}'.format(tmp_path, q))
                    if not img_data is None:
                        X_list.append(img_data)
                        y_list.append(p)
        #print('shape = ', np.array(X_list).shape)
        return np.array(X_list), y_list

    def run(self):
        writers = os.listdir(self.test_path)
        for i, writer in enumerate(writers):
            X = []
            y = []
            tmp_path = '{:s}/{:s}'.format(self.test_path, writer)
            if not os.path.isdir(tmp_path): continue

            test_list = os.listdir(tmp_path)
            for j, img_name in enumerate(test_list):
                #print('=> {:s}/{:s}'.format(tmp_path, img_name))
                img_data = self.read('{:s}/{:s}'.format(tmp_path, img_name))
                if not img_data is None:
                    X.append(img_data)
                    y.append(writer)
                    X_list, y_list = self.get_train_list(img_name[0])
                    self.train(X_list, y_list)
                    result = self.predict(img_data)

                    self.total = self.total + 1
                    if result[0] == writer: self.right = self.right + 1

                    self.reality_list.append(writer)
                    self.predict_list.append(result[0])

    def show(self):
        print('=> right = {:d} total = {:d} right_rate = {:.2f}%'.format(self.right, self.total, float(100 * float(self.right) / float(self.total))))

if __name__ == '__main__':
    o = IMG(debug = True)
    s = svm(debug = True)
    s.run()
    s.show()
