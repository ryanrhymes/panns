Panns -- Nearest Neighbors Search
================================

[![Join the chat at https://gitter.im/ryanrhymes/panns](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ryanrhymes/panns?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


If you do not have patience to read, you can either jump directly to the "Quick Start" section below for example code, or go to the [How-To](https://github.com/ryanrhymes/panns/wiki/How-To) page in panns wiki.

**Note**: the new version of panns stopped storing the actual random vectors in the index in order to achieve even smaller file size. This also means it will not work on the old panns index file. If you still want to use the old version, check 0.1.8 tag or vector branch. I apologize for this inconvenience, but things are always evolving :)

**Follow me on ==> [Twitter](https://twitter.com/ryan_liang),  [Weibo](http://www.weibo.com/olutta),  [Google+](https://www.google.com/+RyanLiang),  [Facebook](http://www.facebook.com/ryan.liang.wang),  [Blogger](http://ryanrhymes.blogspot.com/),  [LinkedIn](http://uk.linkedin.com/in/liangsuomi/)**

## Philosophy

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching [approximate k-nearest neighbors](http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor) in very high dimensional spaces. E.g. one typical use in semantic web is finding the most relevant documents in a big text corpus. Currently, panns supports two distance metrics: Euclidean and Angular (consine). For angular similarity, the dataset need to be normalized. Using panns is straightforward, for example, the following code create two indices.

```python

from panns import *

p1 = PannsIndex(dimension, metric='angular')    # index using cosine distance metric
p2 = PannsIndex(dimension, metric='euclidean')  # index using Euclidean distance metric
...
```

Technically speaking, panns is only a small function module in one of our ongoing projects - [**Kvasir Project**](http://www.cl.cam.ac.uk/~lw525/kvasir/). Kvasir project aims at exploring innovative ways of seamlessly integrating intelligent content provision and recommendation into the web browsing. Kvasir is turning into a more and more interesting project, and I am planning to release an official version by the end of this year. For the details of Kvasir project, please visit our [project website](http://www.cl.cam.ac.uk/~lw525/kvasir/).

The reason that I decided to release panns as a standalone package is simple. During the development of Kvasir, I realized it was actually very difficult to find an scalable tool on the Internet which can perform efficient k-NN search with satisfying accuracy in high dimensional spaces. High dimensionality in this context refers to those datasets having **hundreds of features**, which is already far beyond the capability of standard [k-d tree](http://en.wikipedia.org/wiki/K-d_tree). For the design philosophy, I do not intend to re-invent another super machine learning toolkit which includes everything. Instead, panns has a very clear focus and only tries to do one task well, namely the k-NN search with satisfying accuracy, performance and as small index as possible. It is also worth mentioning that there are many ways to do k-NN in practice. The choice on random projection is based on my consideration on both the ease of understanding (from teaching perspective) and the efficiency of the algorithm (from practical deployment perspective).

panns is developed by Dr. [Liang Wang](http://www.cl.cam.ac.uk/~lw525/) @ Cambridge University, UK. If you have any questions, you can either contact me via email `liang.wang[at]cl.cam.ac.uk` or post in [panns-group](https://groups.google.com/forum/#!forum/panns).



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

panns assumes that the dataset is a row-based matrix (e.g. m x n), where each row represents a data point from a n-dimension feature space. The code snippet below first constructs a 1000 by 100 data matrix, then builds an index of 50 binary trees and saves it to a file. 

Note that you need to explicitly specify the dimensionality in `PannsIndex(...)` before calling `build(...)`, since panns will not try to check your dataset to figure out for you what dimiensionality it actually is.

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

#### Load a dataset

Before indexing your dataset, you can load row vectors one by one into panns using `add_vector(v)`. Variable `v` is supposed to be a numpy row vector.

```python
# 1. load a data point from your dataset
# 2. convert the point to a numpy vector if necessary
# 3. add the vector to panns index
p.add_vector(v)
```

Besides using `add_vector(v)` function, panns supports multiple ways of loading a dataset. For those extremely large datasets, there are two solutions. First, you can use [HDF5](http://www.hdfgroup.org/HDF5/) though the indexing performance will be significantly degraded. Second, you can convert the dataset to a numpy **mmap** matrix then use `load_matrix(A)` to load it directly, which in practice, gives much better performance than the HDF5 solution.

```python
# datasets can be loaded in the following ways
p.load_matrix(A)                     # load a list of row vectors or a numpy matrix
p.load_csv(fname, sep=',')           # load a csv file with specified separator
p.load_hdf5(fname, dataset='panns')  # load a HDF5 file with specified dataset
```

#### Index a dataset (in parallel)

After your dataset is loaded by panns, you can start indexing it by calling `p.build(c)`. Variable `c` specifies the number of trees you want in an index, higher number indicates better accuracy but larger index size. By default, `c = 64` if you do not specify the value explicitly.

```python
# enable the parallel building mode
p.parallelize(True)

# build an index of 128 trees and save to a file
p.build(128)
p.save('test.idx')
```

Usually, building index for a high dimensional dataset can be very time-consuming. panns tries to speed up this process with parallelization. If multiple cores are available, parallel building can be easily enabled with `p.parallelize(True)`. Similarly, you can also disable it by calling `p.parallelize(False)`. By default, parallel mode is **not** enabled.

`p.save('test.idx')` creates two files on your hard disk. One is `test.idx` which stores all the index trees; the other is `test.idx.npy` which is a numpy matrix containing all the raw data vectors of your dataset. For `.npy` file, you can also decide whether to save it as an in-memory file or mmap file.

```python
# save the index as an in-memory file if the raw dataset is small or medium size
# later panns will load the entire .npy file in to the physical memory
p.save('test.idx', mmap=False)

# save the index as mmap file if the raw dataset is huge
# usually, your OS will handle the dynamic loading
p.save('test.idx', mmap=True)
```

#### Load a panns index

Previously generated index file can be loaded by calling `p.load('test.idx')`. Note that panns will automatically look or the file with the name `test.idx.npy` in the same folder of `test.idx`. So, please always put two genereated files together. For loading an index, you do not need to specify the dimensionality of the dataset in `PannsIndex(...)` since such information has been stored in the index file.

```python

from panns import *

p = PannsIndex(metric='euclidean')
p.load('test.idx')

```

#### Query a panns index

The loaded index can be shared among different processes. Therefore, the query performance can be further improved by parallelism. The following code loads the previously generated index file, then performs a simple query. The query returns 10 approximate nearest neighbors.

```python

from panns import *

p = PannsIndex(metric='euclidean')
p.load('test.idx')

v = gaussian_vector(100)
n = p.query(v, 10)
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
| Index Size |       8.6 MB      |       20 MB       |      8.6 MB    |      11 MB     |


Compared with Annoy, panns can achieve higher accuracy with much smaller index file. The reason was actually already briefly discussed in "Theory" section. Generally speaking, the higher accuracy is achieved by placing the offset at sample average; while the smaller index is achieved by reusing the projection vectors.

One thing worth pointing out is the evaluation here is far from thorough and comprehensive, other evaluations are highly welcome and we are always ready to link.



## Discussion

Several toolkits are available for doing the similar job. For example, [spotify/annoy](https://github.com/spotify/annoy), [Flann](http://www.cs.ubc.ca/research/flann/) and [scikit-learn](http://scikit-learn.org/stable/). In terms of balancing the query speed and accuracy, Annoy does an excellent job and performs the best in our tests. Panns, on the other hand, is not really comparable to aforementioned tools regarding the query speed. Though rewriting the code in `C` might help, we will lose the focus and the code will become less accessible for our students. (Python is still a popular option in universities.)

panns scales quite will on the big datasets (with parallel building), and generates smaller indices for extremelly high dimensional datasets. I recommend the following good articles on measureming and comparing different tools, so that you know the pros and cons of each tool:

* [Performance Shootout of Nearest Neighbours](http://radimrehurek.com/2013/11/performance-shootout-of-nearest-neighbours-intro/) by Radim Řehůřek.
* [Annoying blog post](http://erikbern.com/?p=783) by Erik Bernhardsson.

Last comment, I actually would love to see someone who can port the algorithmic logic of panns into annoy which already does an excellent job in k-NN search. Especially, the newest version of panns completely avoids storing the random vectors in indices. I do hope annoy can become the de-facto tool in this small field. Meanwhile, panns will stay simple and serve as elegant starting point for the learners.

Any suggestions, questions and related discussions are warmly welcome. You can post and find relevant information in [panns-group](https://groups.google.com/forum/#!forum/panns) .



## Future Work

* Implement mmap on index file to speed up index loading.
* Improve query performance by parallelism.
* Perform more thorough evaluations.
