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
import mmap
import shutil
import logging
import cPickle as pickle
import multiprocessing

try:
    from scipy import linalg
    import numpy
    import h5py
except Exception, err:
    print 'Warning:', err

from utils import *


logger = logging.getLogger('panns.pannsindex')


class PannsIndex():
    """
    This class builds up the index for the given data set. Two
    metrics are supported: Euclidean and Angular (cosine). The data
    set should be a matrix consisting of row vectors. For cosine, the
    data set is assumed normallized where data poit has length 1.
    """

    def __init__(self, dimension=None, metric='euclidean', dtype='float32'):
        """
        Parameters:
        distance_metric: distance metric to use. euclidean or angular.
        """
        self.dim = dimension    # dimension of data
        self.typ = dtype        # data type of data
        self.mtx = []           # list of row vectors
        self.btr = []           # list of binary-tree
        self.K = 20             # Need to be tweaked
        self.parallel = False

        if metric=='euclidean':
            self.metric = MetricEuclidean()
        elif metric=='angular':
            self.metric = MetricCosine()
        pass


    def add_vector(self, v):
        """
        Add data vector to the current data set.
        """
        self.mtx.append(v)
        # Todo
        # if matrix size is bigger ... then
        pass


    def load_matrix(self, A):
        """
        Load data set from a row-based data matrix.
        """
        self.mtx = A
        pass


    def load_hdf5(self, fname, dataset='panns'):
        """
        Load data set from HDF5 file. Be careful that the performance
        of building up index will significantly degrade due to HDF5
        lookup overheads.

        Parameters:
        fname: file name of the HDF5 data set.
        """
        f = h5py.File(fname, mode='r', driver='core', libver='latest')
        self.mtx = f[dataset]
        pass


    def load_csv(self, fname, sep=','):
        """
        Load data set from a csv file.

        Parameters:
        fname: file name of the csv data set.
        sep:   the separator for the coordinates.
        """
        ### Todo: need to be fixed!
        for line in open(fname, 'r'):
            v = [ numpy.float64(x) for x in line.split(sep) ]
            self.mtx.append(v)
        pass


    def mmap_core_data(self):
        """
        Convert mtx and prj to mmap file to save memory space, very
        useful when dealing with large dataset and parallel mode is
        activated. Skip if the data is already mmaped.
        """
        if type(self.mtx) != numpy.memmap:
            shape_mtx = (len(self.mtx), self.dim)
            mmap_mtx = make_mmap(self.mtx, shape_mtx, self.typ)
            self.mtx = load_mmap(mmap_mtx, shape_mtx, self.typ)
        pass


    def build(self, c=64):
        """
        Build the index for a given data set using random projections.
        The index is a forest of binary trees

        Parameters:
        c: the number of binary trees in the index.
        """
        if self.parallel:
            self.mmap_core_data()
            num_cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(num_cores)
            tbtr = [ pool.apply_async(build_parallel, [self.mtx.filename, self.mtx.shape, self.K, self.typ, t]) for t in xrange(c) ]
            self.btr = [ r.get() for r in tbtr ]
            pool.terminate()
        else:
            self.build_sequential(c)
        pass


    def build_sequential(self, c):
        """
        Build the index sequentially, one projection at a time.

        Parameters:
        c: the number of binary trees to use in building the index.
        """
        for i in xrange(c):
            logger.info('pass %i ...' % i)
            tree = NaiveTree()
            self.btr.append(tree)
            children = range(len(self.mtx))
            self.make_tree(tree.root, children)
        pass


    def make_tree(self, parent, children, lvl=0):
        """
        The actual function which builds up a tree with recursion.

        Parameters:
        parent:   the index of the parent node.
        children: a list of children node indice.
        lvl:      mark the depth of the recursion.
        """
        if len(children) <= max(self.K, lvl):
            parent.nlst = children
            return
        l_child, r_child = None, None
        for attempt in xrange(16):
            parent.proj = numpy.random.randint(2**32-1, dtype=numpy.uint32)
            u = self.random_direction(parent.proj)
            parent.ofst = self.metric.split(u, children, self.mtx)
            l_child, r_child = [], []
            for i in children:
                if self.metric.side(self.mtx[i], u, parent.ofst):
                    r_child.append(i)
                else:
                    l_child.append(i)
            if len(l_child) > 0 and len(r_child) > 0:
                break

        parent.lchd = Node()
        parent.rchd= Node()
        self.make_tree(parent.lchd, l_child, lvl+1)
        self.make_tree(parent.rchd, r_child, lvl+1)
        return


    def query(self, v, c):
        """
        Find the approximate nearest neighbors of a given vector.

        Parameters:
        v: the given vector.
        c: number of nearest neighbors.
        """
        r = set()
        for tree in self.btr:
            idxs = self.get_ann(tree.root, v, c)
            for idx in idxs:
                r.add( (idx, self.metric.distance(self.mtx[idx], v)) )
        r = list(r)
        r.sort(key = lambda x: x[1])
        return r[:c]


    def get_ann(self, p, v, c):
        """
        The function which finds the nearest neighbors recursively.

        Parameters:
        p: the parent node.
        v: the given vector.
        c: number of neighbors.
        """
        nns = None

        if hasattr(p, 'nlst'):
            return p.nlst
        t = numpy.dot(self.random_direction(p.proj), v) - p.ofst
        if t > 0:
            nns = self.get_ann(p.rchd, v, c)
            if len(nns) < c:
                nns += self.get_ann(p.lchd, v, c)
        else:
            nns = self.get_ann(p.lchd, v, c)
            if len(nns) < c:
                nns += self.get_ann(p.rchd, v, c)
        return nns


    def principle_directions(self, c):
        """
        The function returns the principle components of a sample,
        then they are used as projection plane.

        Parameter:
        c: the number of principle components needed.
        """
        B = self.get_samples(max(c,1000))
        u,_,_ = linalg.svd(B.T)
        return u[:,:c].T


    def random_direction(self, seed):
        """
        The function returns a normalized random Gaussian vector
        which is used as the projection plane.

        Parameters:
        seed: random seed for generating the gaussian vector
        """
        return gaussian_vector(self.dim, True, self.typ, seed)


    def get_samples(self, c):
        """
        Get a random sample from the data set.

        Parameters:
        c: number of points in the sample.
        """
        return self.mtx[:c]


    def linear_search(self, v, c):
        """
        Perform a linear search to find the exact nearest neighbors.

        Parameters:
        v: the given vector.
        c: number of nearest neighbors.
        """
        r = list()
        for i in xrange(len(self.mtx)):
            t = ( i, self.metric.distance(self.mtx[i],v) )
            r.append(t)
        r.sort(key = lambda x: x[1])
        return r[:c]


    def save(self, fname='panns.idx', mmap=True):
        """
        The function saves the index in a file using cPickle.

        Parameters:
        fname: the index file name.
        mmap:  enable mmap or not. Enable it if mem is small.
        """
        f = open(fname, 'wb')
        pickle.dump(self.get_basic_info(), f, -1)
        logger.info('dump binary trees to %s ...' % fname)
        for tree in self.btr:
            pickle.dump(tree, f, -1)
        logger.info('dump raw dataset to %s ...' % (fname+'.npy'))
        if mmap:
            make_mmap(self.mtx, (len(self.mtx),self.dim), self.typ, fname+'.npy')
        else:
            numpy.save(open(fname+'.npy', 'wb'), self.mtx)
        pass


    def load(self, fname):
        """
        Load the index into memory, mmap seems not working. :(

        Parameters:
        fname: The index file name.
        """
        f = open(fname, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # step 1, load the basic info
        d = pickle.load(mm)
        self.set_basic_info(d)

        # setp 2, load the binary trees
        logger.info('loading binary trees from %s ...' % fname)
        self.btr = []
        while True:
            try:
                self.btr.append(pickle.load(mm))
            except:
                break

        # step 3, load the raw data set
        logger.info('loading raw dataset from %s ...' % (fname+'.npy'))
        try:
            self.mtx = numpy.load(fname+'.npy')
            logger.info('loading raw dataset as in-mem file ...')
        except Exception, err:
            self.mtx = numpy.memmap(fname+'.npy', dtype=self.typ, mode='r', shape=d['mtx_shape'])
            logger.info('loading raw dataset as mmap file ...')
        pass


    def parallelize(self, enable=False):
        """
        Enable the parallel building, one core for one tree.

        Parameters:
        enable: True is parallel, False is for sequential.
        """
        self.parallel = enable
        pass


    def get_basic_info(self):
        """
        Return a dict containing the basic info of the object.
        """
        d = dict()
        d['mtx_shape'] = (len(self.mtx), self.dim)
        d['typ'] = self.typ
        return d


    def set_basic_info(self, d):
        """
        Set basic properties of the instance for a given dict.
        """
        self.dim = d['mtx_shape'][1]
        self.typ = d['typ']
        pass


    def __del__(self):
        """
        Clean up the temp files and etc.
        """
        tmpdir = tempfile.gettempdir()
        if tmpdir is not None:
            try:
                shutil.rmtree(tmpdir)
            except:
                pass
        pass


    pass
