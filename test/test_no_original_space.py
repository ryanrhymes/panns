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
rows, cols = 10000, 1000
numt = 16


def test_index_file():
    print 'Build a %i x %i dataset using %i trees ...' % (rows, cols, numt)
    vecs = numpy.random.normal(0,1,(rows,cols))

    pidx = PannsIndex(cols, 'angular')
    pidx.load_matrix(vecs)
    pidx.parallelize(True)
    pidx.build(numt)

    pidx.save('mytest.idx', False)
    pass


def test_query_without_original_space():
    # make sure to remove the original space
    # os.rename('mytest.idx.npy', 'mytest.idx.test.npy')

    # start testing query performance and accuracy
    pidx = PannsIndex('angular')
    pidx.load('mytest.idx')

    v = gaussian_vector(cols, True)
    r = pidx.query_without_original_space(v, 20)
    #r = pidx.query(v, 20)
    s = pidx.linear_search(v, 20)

    r = [ y for x, y in r ]
    s = [ x for x, y in s ]
    print r, len(set(r))
    print s, len(set(s))

    accuracy = 1.0 * len(set(r) & set(s)) / len(set(s))
    print 'accuracy = %.6f' % accuracy
    #print 'intersection is ', set(r) & set(s)
    pass


if __name__=='__main__':
    test_index_file()
    test_query_without_original_space()

    sys.exit(0)
