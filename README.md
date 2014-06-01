panns -- Nearest Neighbor Search
====

panns stands for "Python Approximate Nearest Neighbor Search", which is an optimized python library for searching k-nearest neighbors in very high dimensional space. E.g. one tyical use in semantic web is finding the most relevant documents in a big corpus. Currently, panns supports two distance metric: Euclidean and Cosine.


## Quick Start

panns assumes the dataset is a row-based the matrix (e.g. m x n), where each row represents a data point from n-dimension space. The code snippet blow illustrate the basic usage.

```python

from panns import *

p = PannsIndex(f)
for i in xrange(1000):
    p.add_vector()

p.build(50)
```


## Discussion


## Future Work
