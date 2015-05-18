#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Liang Wang @ Computer Lab, Cambridge University
# 2015.05.18
#

import os
import sys
import time
import cPickle as pickle

sys.path.append('../')
from panns import *

import numpy
from scipy import linalg

# experiment settings
rows, cols = 1000, 1000
numt = 1024


def test_index_file():
    print 'Build a %i x %i dataset using %i trees ...' % (rows, cols, numt)
    vecs = numpy.random.normal(0,1,(rows,cols))

    pidx = PannsIndex(cols, 'euclidean')
    pidx.load_matrix(vecs)
    pidx.parallelize(True)
    pidx.build(numt)

    pidx.save('mytest.idx', False)
    pass


def test_query_without_original_space():
    # make sure to remove the original space
    os.rename('mytest.idx.npy', 'mytest.idx.test.npy')

    # start testing query performance and accuracy
    pidx = PannsIndex('euclidean')
    pidx.load('mytest.idx')

    v = gaussian_vector(cols, True)
    r = pidx.query(v, 10)
    pass


if __name__=='__main__':
    test_index_file()
    test_query_without_original_space()

    sys.exit(0)
