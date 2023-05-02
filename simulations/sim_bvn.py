"""
Check speed/accuracy of different BVN approaches, each approach will be benchmarked to 10,000,000 randomly generated values
"""

# External
import os
import numpy as np
# import pandas as pd
# import plotnine as pn
from time import time
from scipy.stats import multivariate_normal as mvn
# Internal
from sntn.dists import bvn
from sntn._bvn import valid_cdf_approach
from parameters import seed
from tests.test_dists_bvn import gen_params

# Growing size of the shapes
n = 1000
params_shape = [(n,), (n, 2), (n, 3), (n, 2, 2), (n, 3, 3), (n, 2, 2, 2), (n, 3, 3, 3), ]
n_params = len(params_shape)

for jj, shape in enumerate(params_shape):
    print(f'--- Shape {jj} of {n_params}: {shape} ---')
    # Draw parameters
    n_params = int(np.prod(shape))
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    # Draw data
    dgp = bvn(mu1, sigma21, mu2, sigma22, rho)
    X = np.squeeze(dgp.rvs(1, seed))
    x1, x2 = np.take(X, 0, -1), np.take(X, 1, -1)
    # Draw data to determine ground truth CDF...
    for ii in np.ndindex(shape):
        mu1[ii], sigma21[ii], mu2[ii], sigma22[ii], rho[ii]
        
    # Get CDF estimate for each approach
    for approach in valid_cdf_approach:
        dist = bvn(mu1, sigma21, mu2, sigma22, rho, approach)
        stime = time()
        pval = dist.cdf(x1=x1, x2=x2)
        dtime = time() - stime

    