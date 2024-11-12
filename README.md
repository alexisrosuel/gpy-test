# gpy-test
Implementation of the Pan-Gao-Yang independence test as described in:
> @article{doi:10.1080/01621459.2013.872037,
author = {Guangming Pan, Jiti Gao and Yanrong Yang},
title = {Testing Independence Among a Large Number of High-Dimensional Random Vectors},
journal = {Journal of the American Statistical Association},
volume = {109},
number = {506},
pages = {600-612},
year = {2014},
publisher = {Taylor & Francis},
doi = {10.1080/01621459.2013.872037},
URL = {     
        https://doi.org/10.1080/01621459.2013.872037
},
eprint = { 
        https://doi.org/10.1080/01621459.2013.872037
}
}


# How to use
```
from gpy_test import GPY
import numpy as np

# get some data, shape is (number of time samples, number of dimensions)
y = np.arange(100*10).reshape(100, 10)

# Choose the test functions
fs = [lambda x: x, lambda x: x**2]

# run the test 
result = GPY(y, fs)

# get the test static and p_value
result.test_statistic, result.p_value
```

Note that this implementation assumes that the time series is complex gaussian, which removes all the terms proportional to E|X|^4 -3. A later improvement to the package could support that.

See notebooks/howto.ipynb for an example of how to use the package, and the other notebooks in the same folder for some assessment of the performance of this test.