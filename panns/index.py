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
import logging
import cPickle as pickle
import multiprocessing

import h5py
import numpy
from scipy import linalg

from utils import *


logger = logging.getLogger('panns.pannsindex')


class PannsIndex():
    """
    This class builds up the index for the given data set. Two
    metrics are supported: Euclidean and Angular (cosine). The data
    set should be a matrix consisting of row vectors. For cosine, the
    data set is assumed normallized where data poit has length 1.
    """

    def __init__(self, dim, distance_metric='euclidean'):
        """
        Parameters:
        distance_metric: distance metric to use. euclidean or angular.
        """
        self.dim = dim    # dimension of data
        self.mtx = []     # list of row vectors
        self.btr = []     # list of binary-tree
        self.prj = []     # list of proj-planes
        self.K = 20       # Need to be tweaked
        self.parallel = False

        if distance_metric=='euclidean':
            self.metric = MetricEuclidean()
        elif distance_metric=='angular':
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
        for line in open(fname, 'r'):
            v = [ numpy.float64(x) for x in line.split(sep) ]
            self.mtx.append(v)
        pass


    def build(self, c=50):
        """
        Build the index for a given data set using random projections.
        The index is a forest of binary trees

        Parameters:
        c: the number of binary trees in the index.
        """
        num_prj = int(2 ** (numpy.log2(len(self.mtx) / self.K) + 1))
        self.prj = self.random_directions(num_prj)
        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(num_cores)
            tbtr = [ pool.apply_async(build_parallel, [self.mtx, self.prj, self.K, t]) for t in xrange(c) ]
            self.btr = [ r.get() for r in tbtr ]
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


    def make_tree(self, parent, children):
        """
        The actual function which builds up a tree with recursion.

        Parameters:
        parent:   the index of the parent node.
        children: a list of children node indice.
        """
        if len(children) <= self.K:
            parent.n_list = children
            return
        parent.projection = numpy.random.randint(len(self.prj))
        u = self.prj[parent.projection]
        parent.offset = self.metric.split(u, children, self.mtx)
        l_child, r_child = [], []
        for i in children:
            if self.metric.side(self.mtx[i], u, parent.offset):
                r_child.append(i)
            else:
                l_child.append(i)
        parent.l_child = Node()
        parent.r_child = Node()
        self.make_tree(parent.l_child, l_child)
        self.make_tree(parent.r_child, r_child)
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
                r.add( (self.metric.distance(self.mtx[idx], v), idx) )
        r = list(r)
        r.sort()
        return [ x for _, x in r[:c]]


    def get_ann(self, p, v, c):
        """
        The function which finds the nearest neighbors recursively.

        Parameters:
        p: the parent node.
        v: the given vector.
        c: number of neighbors.
        """
        nns = None
        if p.n_list is not None:
            return p.n_list
        t = numpy.dot(self.prj[p.projection], v) - p.offset
        if t > 0:
            nns = self.get_ann(p.r_child, v, c)
            if len(nns) < c:
                nns += self.get_ann(p.l_child, v, c)
        else:
            nns = self.get_ann(p.l_child, v, c)
            if len(nns) < c:
                nns += self.get_ann(p.r_child, v, c)
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


    def random_directions(self, c):
        """
        The function returns Gaussian random directions which are
        used as projection plane.

        Parameters:
        c: the number of principle components needed.
        """
        return [ gaussian_vector(self.dim, True) for _ in xrange(c) ]


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
            t = ( self.metric.distance(self.mtx[i],v), i )
            r.append(t)
        r.sort()
        return [ x for _, x in r[:c] ]


    def save(self, fname='panns.idx'):
        """
        The function saves the index in a file using cPickle.

        Parameters:
        fname: the index file name.
        """
        f = open(fname, 'wb')
        logger.info('dump binary trees ...')
        pickle.dump(self.btr, f, -1)
        logger.info('dump random vectors ...')
        pickle.dump(self.prj, f, -1)
        pass


    def load(self, fname):
        """
        Load the index into memory, mmap seems not working. :(

        Parameters:
        fname: The index file name.
        """
        f = open(fname, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        logger.info('load binary trees ...')
        self.btr = pickle.load(mm)
        logger.info('load random vectors ...')
        self.prj = pickle.load(mm)
        pass


    def parallelize(self, enable=False):
        """
        Enable the parallel building, one core for one tree.

        Parameters:
        enable: True is parallel, False is for sequential.
        """
        self.parallel = enable
        pass


    # Todo: Either remove or enhance
    def dump_rawdata(self, fname, vecs):
        """
        index dumping and loading needs improvements.
        """
        f = open(fname, 'wb')
        for v in vecs:
            f.write(pickle.dumps(v, -1))
        f.close()
        pass


    pass
