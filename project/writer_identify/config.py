#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/05/01, homeway'

import os

class Config(object):
    base = os.path.dirname(os.path.realpath(__file__))
    test_path = '{:s}/{:s}'.format(base, 'test')
    train_path = '{:s}/{:s}'.format(base, 'train')
    source_path = '{:s}/{:s}'.format(base, 'source')