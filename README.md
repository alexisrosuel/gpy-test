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

# get some data, shape is (number of time samples, number of dimensions)
y = np.array(...)

# define the gpy config 
from gpy_test import GPY, CovarianceConfig

d = {
    "integral_config": {
        "epsabs": 1e-2,
        "epsrel": 1e-2,
    },
    "contour_pair_config": {
        "contours": [
            {"real_slack": 1, "type_": "circle"},
            {"real_slack": 2, "type_": "circle"},
        ]
    },
    "admissible_imag": 1e-3,
    "admissible_negative": 1e2,
    "n_jobs": 1,
    "verbose": False,
}
covariance_config = CovarianceConfig(**d)

# Choose the test functions
fs = [lambda x: x, lambda x: x**2]

# run the test 
result = GPY(y, fs, covariance_config=covariance_config)

# get the test static and p_value
result.test_statistic, result.p_value
```

Note that this implementation assumes that the time series is complex gaussian, which removes all the terms proportional to E|X|^4 -3. A later improvement to the package could support that.


See notebooks/howto.ipynb for an example of how to use the package 

See also notebooks/leverl_power_check.ipynb for a study of the performance of this test.