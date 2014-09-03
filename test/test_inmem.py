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


def test_index_file():
    rows, cols = 1000, 100
    print 'Build a %i x %i dataset ...' % (rows, cols)
    vecs = numpy.random.normal(0,1,(rows,cols))

    pidx = PannsIndex(cols, 'euclidean')
    pidx.load_matrix(vecs)
    pidx.parallelize(True)
    pidx.build(16)

    t = time.time()
    v = gaussian_vector(cols, True)

    t = time.time()
    print 'Test saving the index files ...'
    pidx.save('mytest.idx', False)
    print 'Saving is done in %.3f s' % (time.time()-t)

    print 'Test loading the index and querying ...'
    t = time.time()
    pidx.load('mytest.idx')
    print 'Loading is done in %.3f s' % (time.time()-t)

    t = time.time()
    r = pidx.query(v, 10)
    print 'Test querying done in %.3f s' % (time.time()-t)
    pass


if __name__=='__main__':
    test_index_file()

    sys.exit(0)
