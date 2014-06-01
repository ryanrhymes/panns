panns -- Nearest Neighbor Search
====

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching k-nearest neighbors in very high dimensional spaces. E.g. one tyical use in semantic web is finding the most relevant documents in a big corpus of text. Currently, panns supports two distance metric: Euclidean and Cosine.


## Quick Start

panns assumes the dataset is a row-based the matrix (e.g. m x n), where each row represents a data point from an n-dimension space. The code snippet blow illustrate the basic usage.

```python

from panns import *

p = PannsIndex('euclidean')
for i in xrange(1000):
    v = gaussian_vector(100)
    p.add_vector(v)
p.build(50)

p.save('test.idx')
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

p = PannsIndex('euclidean')

....

p.parallelize(True)
p.build(100)

```



## Discussion

Any suggestions, questions and related discussions are welcome and can be found here https://groups.google.com/forum/#!forum/panns

## Future Work
