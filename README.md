panns -- Nearest Neighbor Search
====

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching k-nearest neighbors in very high dimensional spaces. E.g. one tyical use in semantic web is finding the most relevant documents in a big corpus of text. Currently, panns supports two distance metric: Euclidean and Cosine.


Features:

* Support raw, CSV and HDF5 datasets.
* Support parallel building of indices.
* Generate smaller index file than other libraries.
* Achieve higher accuracy.


## Installation

Algebra operations in panns rely on both Numpy and Scipy, please make sure you have these two packages properly installed before using panns. The installation can be done by the following shell commands.

```bash
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade
```


The installation of panns is very straightforward. You can either install it directly from PyPI (probably the easiest way), or download the source code then install manually.
```bash
sudo pip install panns --upgrade
```


If you are interested in the source code or even want to contribute to make it faster and better, you can clone the code from Github.
```bash
git clone git@github.com:ryanrhymes/panns.git
```


## Quick Start

panns assumes that the dataset is a row-based the matrix (e.g. m x n), where each row represents a data point from an n-dimension feature space. The code snippet below first constructs a 1000 by 100 data matrix, then builds an index of 50 binary trees and saves it to a file.

```python

from panns import *

p = PannsIndex('euclidean')
for i in xrange(1000):
    v = gaussian_vector(100)
    p.add_vector(v)
p.build(50)

p.save('test.idx')
```

Besides using "add_vector(v)" function, panns supports multiple ways of loading a dataset. For extremely large dataset, HDF5 is recommended though the building performance will be significantly degraded.

```python

p = PannsIndex('angular')
p.load_matrix()
p.load_csv()
p.load_hdf5()
```


The following code loads the previously generated index file, then perform a simple query. The query returns 10 approximate nearest neighbors.

```python

from panns import *

p = PannsIndex('euclidean')
p.load('test.idx')

v = gaussian_vector(100)
n = p.query(v, 10)
```


Usually, building index for a high dimensional dataset can be very time-consuming. panns tries to speed up this process by optimizing the code and taking advantage of physical resources. If multiple cores are available, parallel building can be easily enabled as follows:

```python

from panns import *

p = PannsIndex('angular')

....

p.parallelize(True)
p.build(100)

```


## Theory In a Nutshell

Random projection



## Discussion

Any suggestions, questions and related discussions are welcome and can be found in [panns-group](https://groups.google.com/forum/#!forum/panns) .

## Future Work
