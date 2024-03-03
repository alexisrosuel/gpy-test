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
y = ... 

# define the gpy config 
from gpy_test import GPY, CovarianceConfig

d = {
    "integral_config": {
        "type_": "dblquad",
        "epsabs": 1e-2,
        "epsrel": 1e-2,
    },
    "fixed_point_config": {
        "init_m_real": 1.0,
        "init_m_imag": 1.0,
        "max_steps": 100,
        "tolerance": 1e-4,
    },
    "contour_config_pair": (
        {"real_slack": 0, "type_": "circle"},
        {"real_slack": 0.5, "type_": "circle"},
    ),
    "derivative_epsilon": 1e-08,
    "admissible_imag": 1e-3,
    "n_jobs": 1,
    "verbose": False,
}
covariance_config = CovarianceConfig(**d)

# Choose the test functions
fs = [lambda x: x, lambda x: x**2]

# run the test 
result = GPY(y, fs, covariance_config=covariance_config)

# get the p_value
result.p_value
```

## Other usage

It is also possible to define a spectral density to be used. Can be either a callable, or a pair of (frequencies, spectral density values)

```
oracle_sd = lambda nu: ...
result = GPY(y, fs, covariance_config=covariance_config, sd=sd)
```

or 
```
freqs, spectral_density_values = ...
result = GPY(y, fs, covariance_config=covariance_config, sd=(freqs, spectral_density_values))
```

Lastly, it is also possible to compute upfront the limit covariance used in the test, and pass it to the test: 
```
eig_range = (-2, 10) # contains the support of the limiting distribution of the sample covariance matrix
Cov = compute_covariance(
        covariance_config,
        fs,
        eig_range,
        c=M / N,
        sd=sd
        )
result = GPY(y, fs, Cov)
```

Note that this implementation assumes that the time series is complex gaussian, which removes all the terms proportional to E|X|^4 -3. A later improvement to the package could support that.


See notebooks/howto.ipynb for an example of how to use the package 

See also notebooks/level.ipynb and power.ipynb for a study of the performance of this test.