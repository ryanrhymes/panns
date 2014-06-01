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
from annoy import *

import numpy
from scipy import linalg


def random_norm_row_matrix(m, n, seed=None):
    if seed is not None:
        numpy.random.seed(seed)
    vecs = numpy.random.normal(0,1,(m,n))
    for i in xrange(m):
        vecs[i,:] /= linalg.norm(vecs[i,:])
    return vecs


def test_annoy(annoy, v, ans):
    t = time.time()
    n = annoy.get_nns_by_vector(v.tolist(), 10)
    r1 = precision(ans, n)
    r2 = recall(ans, n)
    print "Annoy precision: %.3f\t recall: %.3f\t %.4f ms" % (r1, r2, (time.time()-t)*1000)
    return r1, r2


def test_panns(panns, v, ans):
    t = time.time()
    n = panns.query(v, 10)
    r1 = precision(ans, n)
    r2 = recall(ans, n)
    print "Panns precision: %.3f\t recall: %.3f\t %.4f ms" % (r1, r2, (time.time()-t)*1000)
    return r1, r2


def compare_both(metric):
    m, n = 5000, 200
    print 'Build a %i x %i dataset ...' % (m, n)
    vecs = random_norm_row_matrix(m, n) # + 5.5

    num_prj = 128

    t = AnnoyIndex(n, metric)
    for i in xrange(m):
        t.add_item(i, vecs[i].tolist())
    t.build(num_prj)

    p = PannsIndex(n, metric)
    p.load_matrix(vecs)
    p.parallelize(True)
    p.build(num_prj)

    r1,r2,r3,r4 = 0,0,0,0
    for i in range(50):
        print '+'*30, 'test', i
        v = gaussian_vector(n, True)
        ans = p.linear_search(v, 10)

        t1, t2 = test_annoy(t, v, ans)
        t3, t4 = test_panns(p, v, ans)
        r1 += t1
        r2 += t2
        r3 += t3
        r4 += t4

    print '='*30, 'Summary', metric
    print "Annoy precision: %.3f\t recall: %.3f" % (r1/50, r2/50)
    print "Panns precision: %.3f\t recall: %.3f" % (r3/50, r4/50)

    t.save('annoy.'+metric+'.idx')
    p.save('panns.'+metric+'.idx')
    pass



if __name__=='__main__':
    compare_both('euclidean')
    compare_both('angular')
    sys.exit(0)
