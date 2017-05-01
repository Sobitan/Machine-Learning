#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/01, homeway'

from config import Config as conf
import os, random

class Generate(object):
    def __init__(self, config = conf, split = 0.3, debug = False):
        self.conf = conf
        self.split = split
        self.debug = debug

    def build(self):
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

if __name__ == '__main__':
    G = Generate(split = 0.2)
    G.recover().build()