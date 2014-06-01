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
from scipy import linalg
from scipy.spatial import distance


logger = logging.getLogger('panns.utils')


class Node():
    projection = None
    offset = None
    l_child = None
    r_child = None
    n_list = None
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
        v = numpy.zeros(len(u))
        for i in idxs:
            v += mtx[i]
        v = v / len(idxs)
        a = numpy.dot(u, v)
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
        return numpy.dot(u, v) - offset > 0

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


def gaussian_vector(size, normalize=False):
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


def build_parallel(mtx, prj, K, t):
    """
    The function for parallel building index. Implemented here because
    the default python serialization cannot pickle instance function.

    Parameters:
    mtx: a row-based data set.
    K:   max number of data points on a leaf.
    t:   index of binary trees.
    """
    logger.info('pass %i ...' % t)
    numpy.random.seed(t**2)
    tree = NaiveTree()
    children = range(len(mtx))
    make_tree_parallel(tree.root, children, mtx, prj, K)
    return tree


def make_tree_parallel(parent, children, mtx, prj, K):
    """
    Builds up a binary tree recursively, for parallel building.

    Parameters:
    parent:   parent node index.
    children: a list of children node indices.
    mtx:      a row-based data set.
    K:        max number of data points on a leaf.
    """
    if len(children) <= K:
        parent.n_list = children
        return
    parent.projection = numpy.random.randint(len(prj))
    u = prj[parent.projection]
    parent.offset = Metric.split(u, children, mtx)
    l_child, r_child = [], []
    for i in children:
        if Metric.side(mtx[i], u, parent.offset):
            r_child.append(i)
        else:
            l_child.append(i)
    parent.l_child = Node()
    parent.r_child = Node()
    make_tree_parallel(parent.l_child, l_child, mtx, prj, K)
    make_tree_parallel(parent.r_child, r_child, mtx, prj, K)
    return
