#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Liang Wang <liang.wang@cs.helsinki.fi>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Liang Wang @ CS Dept, Helsinki Univ, Finland
# 2014.05.01
#


import logging
import numpy
import tempfile
from scipy import linalg
from scipy.spatial import distance


logger = logging.getLogger('panns.utils')


class Node():
    __slots__ = ['proj', 'ofst', 'lchd', 'rchd', 'nlst']
    pass


class NaiveTree(object):

    def __init__(self):
        self.root = Node()
        pass

    pass


class Metric():
    """
    The basic metric class used in Panns index building. Super class
    of MetricEuclidean and MetricCosine.
    """

    @staticmethod
    def split(u, idxs, mtx):
        """
        Project the data points on a random vector and return the
        average value.

        Parameters:
        u:    random vector.
        idxs: data points to project.
        mtx:  data set.
        """
        v = numpy.zeros(len(u), u.dtype)
        for i in idxs:
            v += mtx[i]
        a = numpy.dot(u, v) / len(idxs)
        return a


    @staticmethod
    def side(u, v, offset):
        """
        Project v on u then check which side it falls in given the
        offset.

        Parameters:
        u: random vector.
        v: data point to project.
        """
        r = None
        x = numpy.dot(u, v) - offset
        if abs(x) < 1e-12:
            r = ( numpy.random.uniform(0,1,1)[0] > 0.5 )
        else:
            r = ( x > 0 )
        return r

    pass


class MetricEuclidean(Metric):
    """
    Metric class for Euclidean index.
    """

    @staticmethod
    def distance(u, v):
        return distance.euclidean(u, v)

    pass


class MetricCosine(Metric):
    """
    Metric class for cosine index.
    """

    @staticmethod
    def distance(u, v):
        return 1.0 - numpy.dot(u, v)

    pass


def gaussian_vector(size, normalize=False, dtype='float32'):
    """
    Returns a (normalized) Gaussian random vector.

    Parameters:
    normalize: the vector length is normalized to 1 if True.
    """
    v = numpy.random.normal(0,1,size)
    if normalize:
        v = v / linalg.norm(v)
    return v


def precision(relevant, retrieved):
    """
    Return the precision of the search result.

    Parameters:
    relevant:  the relevant data points.
    retrieved: the retireved data points
    """
    r = 1.0 * len(set(relevant) & set(retrieved)) / len(retrieved)
    return r


def recall(relevant, retrieved):
    """
    Return the recall of the search result.

    Parameters:
    relevant:  the relevant data points.
    retrieved: the retireved data points
    """
    r = 1.0 * len(set(relevant) & set(retrieved)) / len(relevant)
    return r


def build_parallel(mtx, prj, shape_mtx, shape_prj, K, dtype, t):
    """
    The function for parallel building index. Implemented here because
    the default python serialization cannot pickle instance function.

    Parameters:
    mtx: a row-based data set, should be an numpy matrix.
    K:   max number of data points on a leaf.
    t:   index of binary trees.
    """
    logger.info('pass %i ...' % t)
    mtx = numpy.memmap(mtx, dtype=dtype, mode='r', shape=shape_mtx)
    prj = numpy.memmap(prj, dtype=dtype, mode='r', shape=shape_prj)
    numpy.random.seed(t**2)
    tree = NaiveTree()
    children = range(len(mtx))
    make_tree_parallel(tree.root, children, mtx, prj, K)
    return tree


def make_tree_parallel(parent, children, mtx, prj, K, lvl=0):
    """
    Builds up a binary tree recursively, for parallel building.

    Parameters:
    parent:   parent node index.
    children: a list of children node indices.
    mtx:      a row-based data set.
    K:        max number of data points on a leaf.
    """
    if len(children) <= max(K, lvl):
        parent.nlst = children
        return
    l_child, r_child = None, None
    for attempt in xrange(16):
        parent.proj = numpy.random.randint(len(prj))
        u = prj[parent.proj]
        parent.ofst = Metric.split(u, children, mtx)
        l_child, r_child = [], []
        for i in children:
            if Metric.side(mtx[i], u, parent.ofst):
                r_child.append(i)
            else:
                l_child.append(i)
        if len(l_child) > 0 and len(r_child) > 0:
                break
    parent.lchd = Node()
    parent.rchd = Node()
    make_tree_parallel(parent.lchd, l_child, mtx, prj, K)
    make_tree_parallel(parent.rchd, r_child, mtx, prj, K)
    return


def make_mmap(mtx, shape, dtype, fname=None):
    m, n  = shape
    if fname is None:
        fname = tempfile.mkstemp()[1]
    logger.info('mmaping the data to %s ...' % fname)
    fpw = numpy.memmap(fname, dtype=dtype, mode='w+', shape=(m,n))
    for i in xrange(m):
        fpw[i] = mtx[i]
    del fpw
    return fname


def load_mmap(fname, shape, dtype):
    mtx = numpy.memmap(fname, dtype=dtype, mode='r', shape=shape)
    return mtx
