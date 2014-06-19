#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Liang Wang <liang.wang@cs.helsinki.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.06.19
#


import os
import sys
import time

sys.path.append('../')
from panns import *

import numpy
from scipy import linalg


def test_big_data():
    rows, cols = 10000, 1000
    logger.info('start building %i x %i matrix ...' % (rows, cols))
    vecs = numpy.random.normal(0,1,(rows,cols))
    pidx = PannsIndex(cols, 'euclidean')
    logger.info('finish building  %i x %i matrix.' % (rows, cols))
    pidx.load_matrix(vecs)

    ntrees = 16
    logger.info('start building the index ...')
    pidx.parallelize(True)
    pidx.build(ntrees)
    logger.info('finish buliding the index.')

    logger.info('save index file ...')
    pidx.save('mytest.idx')
    pass


if __name__=='__main__':
    test_big_data()

    sys.exit(0)
