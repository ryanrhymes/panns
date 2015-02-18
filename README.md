panns -- Nearest Neighbors Search
================================


![Downloads](https://pypip.in/d/panns/badge.png "Downloads") . ![License](https://pypip.in/license/gensim/badge.png "License")

If you do not have patience to read, you can either directly jump to the "Quick Start" section below for example code, or go to the [How-To](https://github.com/ryanrhymes/panns/wiki/How-To) page in panns wiki.



## Background

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching [approximate k-nearest neighbors](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor) in very high dimensional spaces. E.g. one typical use in semantic web is finding the most relevant documents in a big text corpus. Currently, panns supports two distance metrics: Euclidean and Angular (consine). For angular similarity, the dataset need to be normalized. Using panns is straightforward, for example, the following code create two indices.

```python

from panns import *

p1 = PannsIndex(metric='angular')    # index using cosine distance metric
p2 = PannsIndex(metric='euclidean')  # index using Euclidean distance metric
...
```

Technically speaking, panns is only a small function module in one of our ongoing projects - [**Kvasir Project**](https://www.cs.helsinki.fi/liang.wang/kvasir). Kvasir project aims at exploring innovative ways of seamlessly integrating intelligent content provision and recommendation into the web browsing. Kvasir is turning into a more and more interesting project, and I am planning to release an official version by the end of this year. For the details of Kvasir project, please visit our [project website](https://www.cs.helsinki.fi/liang.wang/kvasir).

The reason that I decided to release panns as a standalone package is simple. During the development of Kvasir, I realized it was actually very difficult to find an easy-to-use tool on the Internet which can perform efficient k-NN search with satisfying accuracy in high dimensional spaces. High dimensionality in this context refers to those datasets having **hundreds of features**, which is already far beyond the capability of standard [k-d tree](http://en.wikipedia.org/wiki/K-d_tree). For the design philosophy, I do not intend to re-invent another super machine learning toolkit which includes everything but good at nothing. Instead, panns has a very clear focus and only tries to do one task well, namely finding approximate nearest neighbors for a given point with reasonable speed and small size index. It is also worth mentioning that there are many ways to do k-NN in practice. The choice on random projection is based on my consideration on both the ease of understanding (from teaching perspective) and the efficiency of the algorithm (from practical deployment perspective). Still, panns can easily beat many similar tools at the moment.

panns is developed by Dr. [Liang Wang](http://cs.helsinki.fi/liang.wang) @ Cambridge University, UK. If you have any questions, you can either contact me via email `liang.wang[at]helsinki.fi` or post in [panns-group](https://groups.google.com/forum/#!forum/panns).



## Features

* Pure python implementation.
* Optimized for large and high-dimension dataset (e.g. > 500).
* Generate small index file with high query accuracy.
* Support both Euclidean and cosine distance metrics.
* Support parallel building of indices.
* Small memory usage and index can be shared among processes.
* Support raw, csv, numpy and [HDF5](http://www.hdfgroup.org/HDF5/) datasets.



## Installation

Algebra operations in panns rely on both [Numpy](http://www.numpy.org/) and [Scipy](http://www.scipy.org/), and HDF5 operations rely on [h5py](http://www.h5py.org/). Note h5py is optional if you do not need operate on HDF5 files. Please make sure you have these packages properly installed before using the full features of panns. The installation can be done by the following shell commands.

```bash
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade
sudo pip install h5py --upgrade
```


The installation of panns is very straightforward. You can either install it directly from PyPI (probably the easiest way), or download the source code then install manually.
```bash
sudo pip install panns --upgrade
```


If you are interested in the source code or even want to contribute to make it faster and better, which I would highly appreciate, you can clone the code from Github.
```bash
git clone git@github.com:ryanrhymes/panns.git
```



## Quick Start

### Create a panns instance

panns assumes that the dataset is a row-based the matrix (e.g. m x n), where each row represents a data point from an n-dimension feature space. The code snippet below first constructs a 1000 by 100 data matrix, then builds an index of 50 binary trees and saves it to a file.

```python

from panns import *

# create an index of Euclidean distance
p = PannsIndex(dimension=100, metric='euclidean')

# generate a 1000 x 100 dataset
for i in xrange(1000):
    v = gaussian_vector(100)
    p.add_vector(v)

# build an index of 50 trees and save to a file
p.build(50)
p.save('test.idx')
```

Besides using `add_vector(v)` function, panns supports multiple ways of loading a dataset. For those extremely large datasets, [HDF5](http://www.hdfgroup.org/HDF5/) is recommended though the building performance will be significantly degraded. However, the performance can be improved by enabling parallel building as shown later.

```python
# datasets can be loaded in the following ways
p.load_matrix(A)                     # load a list of row vectors or a numpy matrix
p.load_csv(fname, sep=',')           # load a csv file with specified separator
p.load_hdf5(fname, dataset='panns')  # load a HDF5 file with specified dataset
```

The saved index can be loaded and shared among different processes for future use. Therefore, the query performance can be further improved by parallelism. The following code loads the previously generated index file, then performs a simple query. The query returns 10 approximate nearest neighbors.

```python

from panns import *

p = PannsIndex(metric='euclidean')
p.load('test.idx')

v = gaussian_vector(100)
n = p.query(v, 10)
```


Usually, building index for a high dimensional dataset can be very time-consuming. panns tries to speed up this process from two perspectives: optimizing the code and taking advantage of the physical resources. If multiple cores are available, parallel building can be easily enabled as follows:

```python

from panns import *

p = PannsIndex(metric='angular')

....

p.parallelize(True)
p.build()

```



## Theory In a Nutshell

Simply put, approximate k-NN in panns is achieved by [random projection](http://en.wikipedia.org/wiki/Locality-sensitive_hashing#Random_projection). The index is built by constructing a binary tree. Each node of the tree represents a scalar-projection of certain data points, which are further divided into two groups (left- and right-child) by comparing to their average. The accuracy can be improved from the following perspective:

* Place the offset wisely (e.g. at the sample average).
* Choose the projection vector wisely (e.g. random or principle components).
* Use more projections (but longer building time and larger index).
* Use more binary trees (also longer building time and larger index).

The accuracy of approximate k-NN is usually achieved at the price of large index. panns aims to find the good trade-off of these two conflicting factors. Different from other libraries, panns reuses the projection vectors among different trees instead of generating a new random vector for each node. This can significantly reduces the index size when the dimension is high and trees are many. At the same time, reusing the projection vectors will not degrade the accuracy (see Evaluation section below).



## Evaluation

Evaluation in this section is simply done by comparing against Annoy. Annoy is a C++ implementation of similar functionality as panns, it is used in Spotify recommender system. In the evaluation, we used a 5000 x 200 dataset, namely 5000 200-dimension feature vectors. For fair comparison, both Annoy and panns use 128 binary trees, and evaluation was done with two distance metrics (Euclidean and cosine). The following table summarizes the results. (data type?)

|            | panns (Euclidean) | Annoy (Euclidean) | panns (cosine) | Annoy (cosine) |
|:----------:|:-----------------:|:-----------------:|:--------------:|:--------------:|
|  Accuracy  |       69.2%       |       48.8%       |      70.1%     |      50.4%     |
| Index Size |       5.4 MB      |       20 MB       |     5.4 MB     |      11 MB     |


Compared with Annoy, panns can achieve higher accuracy with much smaller index file. The reason was actually already briefly discussed in "Theory" section. Generally speaking, the higher accuracy is achieved by placing the offset at sample average; while the smaller index is achieved by reusing the projection vectors.

One thing worth pointing out is the evaluation here is far from thorough and comprehensive, other evaluations are highly welcome and we are always ready to link.



## Discussion

Any suggestions, questions and related discussions are warmly welcome. You can post and find relevant information in [panns-group](https://groups.google.com/forum/#!forum/panns) .



## Future Work

* Implement mmap on index file to speed up index loading.
* Improve query performance by parallelism.
* Perform more thorough evaluations.
