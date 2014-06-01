#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.05.20
#

import os
import sys
import time

sys.path.append('../')
from panns import *

import h5py
import numpy


def test_iterate_performance():
    t = time.time()
    m, n = 10000, 500
    mtx = numpy.random.normal(0, 1, (m, n))
    print 'Matrix is generated, spent %.3f s' % (time.time()-t)

    f = h5py.File('test.hdf5', mode='w', driver='core', libver='latest')
    dset = f.create_dataset('panns', (m, n), dtype='float')
    t = time.time()
    for i in xrange(m):
        dset[i,:] = mtx[i,:]
    f.close()
    print 'Iteration-based update is done in %.3f s' % (time.time()-t)
    pass


def test_block_performance():
    t = time.time()
    m, n = 10000, 500
    mtx = numpy.random.normal(0, 1, (m, n))
    print 'Matrix is generated, spent %.3f s' % (time.time()-t)

    f = h5py.File('test.hdf5', mode='w', driver='core', libver='latest')
    dset = f.create_dataset('panns', (m, n), dtype='float64')
    t = time.time()
    dset[:] = mtx[:]
    f.close()
    print 'Block-based update is done in %.3f s' % (time.time()-t)
    pass


def test_lookup_performance(fn):
    f = h5py.File(fn, mode='r', driver='core', libver='latest')
    dset = f['panns']
    m, n = dset.shape
    z = numpy.zeros((1,n))
    t0, c = 0.0, 100000
    for i in xrange(c):
        t1 = time.time()
        idx = numpy.random.randint(m)
        z = dset[idx,:]
        t0 += time.time() - t1
    print 'Average lookup time is %.2f ms' % (t0 * 1000 / c)
    pass


def test_building_index(fn):
    p = PannsIndex(500, 'euclidean')

    mtx = numpy.random.normal(0,1,(10000,500))
    p.load_matrix(mtx)
    t = time.time()
    p.build(5)
    print 'Mem-based building took %.3f s' % (time.time()-t)

    p.load_hdf5(fn)
    t = time.time()
    p.build(5)
    print 'HDF5-based building took %.3f s' % (time.time()-t)
    pass


if __name__=='__main__':
    test_iterate_performance()
    #test_block_performance()
    #test_lookup_performance('test.hdf5')
    test_building_index('test.hdf5')
    sys.exit(0)
