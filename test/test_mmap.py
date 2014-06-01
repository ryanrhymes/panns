#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.05.20
#

import os
import sys
import time
import cPickle as pickle

sys.path.append('../')
from panns import *

import numpy
from scipy import linalg


def generate_large_matrix():
    rows, cols = 1000000, 500
    print 'Test serializing a %i x %i matrix ...' % (rows, cols)
    t = time.time()
    vecs = numpy.random.normal(0,1,(rows,cols))
    print 'Matrix constructed, spent %.2f s' % (time.time() - t)

    f1 = open('test_data1', 'wb')
    t = time.time()
    print 'saving as numpy npz format ...'
    numpy.savez_compressed(f1, vecs)
    print 'save done, spent %.2f s' % (time.time() - t)
    f1.close()

    f2 = open('test_data2', 'wb')
    t = time.time()
    print 'saving as self-defined format ...'
    for v in vecs:
        f2.write(pickle.dumps(v, -1))
    f2.close()
    print 'save done, spent %.2f s' % (time.time() - t)
    pass


def test_load_matrix(fn):
    print 'Loading matrix from self-defined format ...'
    f = open(fn, 'rb')
    mtx = []
    t = time.time()
    while True:
        try:
            v = pickle.load(f)
            mtx.append(v)
        except:
            break
    print 'The matrix loaded from self-defined format, spent %.2f s' % (time.time() - t)
    pass



if __name__=='__main__':
    generate_large_matrix()
    test_load_matrix('test_data2')
    sys.exit(0)
