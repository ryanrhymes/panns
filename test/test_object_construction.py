#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Liang Wang <liang.wang@cs.helsinki.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.05.01
#


import os
import sys
import time

sys.path.append('../')
from panns import *

import numpy
from scipy import linalg


def test_add_vector():
    rows, cols = 50000, 50
    vecs = numpy.random.normal(0,1,(rows,cols))
    pidx = PannsIndex(cols, 'euclidean')
    print 'Test adding vector function by building %i x %i dataset ...' % (rows, cols)
    t = time.time()
    for v in vecs:
        pidx.add_vector(v)
    print 'Test is done in %.3f s' % (time.time() - t)
    pass


def test_index_file():
    rows, cols = 10000, 500
    print 'Build a %i x %i dataset ...' % (rows, cols)
    vecs = numpy.random.normal(0,1,(rows,cols))

    pidx = PannsIndex(cols, 'euclidean')
    pidx.load_matrix(vecs)
    pidx.parallelize(False)
    pidx.build(128)

    t = time.time()
    v = gaussian_vector(cols, True)
    r1 = pidx.linear_search(v, 10)
    print 'Linear search is done in %.3f s' % (time.time()-t)
    r2 = pidx.query(v, 10)
    m1 = precision(r1, r2)
    m2 = recall(r1, r2)
    print "Precision: %.3f, Recall: %.3f" % (m1, m2)

    t = time.time()
    print 'Test saving the index files ...'
    pidx.save('mytest.idx')
    print 'Saving is done in %.3f s' % (time.time()-t)

    print 'Test loading the index and querying ...'
    t = time.time()
    pidx.load('mytest.idx')
    print 'Loading is done in %.3f s' % (time.time()-t)

    t = time.time()
    r = pidx.query(v, 10)
    print 'Test querying done in %.3f s' % (time.time()-t)
    pass


def test_parallel_build():
    print 'Test building index in parallel mode ...'
    rows, cols = 100000, 50
    print 'Build a %i x %i dataset ...' % (rows, cols)
    vecs = numpy.random.normal(0,1,(rows,cols))

    pidx = PannsIndex(cols)
    pidx.load_matrix(vecs)
    pidx.parallelize(True)
    pidx.build(100)
    print 'Parallel bulding is done.'
    v = gaussian_vector(cols, True)
    r1 = pidx.linear_search(v, 10)
    r2 = pidx.query(v, 10)
    m1 = precision(r1, r2)
    m2 = recall(r1, r2)
    print "Precision: %.3f, Recall: %.3f" % (m1, m2)
    pass


def test_load_csv(fname):
    pidx = PannsIndex(50)
    pidx.load_csv(fname, ',')
    print pidx.mtx
    pass


if __name__=='__main__':
    ###test_add_vector()
    test_index_file()
    ###test_parallel_build()
    ###test_load_csv('zzz')
    sys.exit(0)
