## `sntn`: Sum of Normal and Truncated Normal package

This repo is for the `sntn` package. Reference to arxiv.

# Installation

see pypi

# Toy examples

## (i) Fast bivariate CDF

## (ii) Filtering regime

Suppose an industrial process combines two parts together, but the first part is only kept if it's length exceeds a certain value. According to existing guidelines that first piece should be N(50,3) and if only kepy if >=50, and the second piece should be N(70,5). The code block below tests the power of the `SNTN` class to detect whether this distribution actually conforms to our null distribution: $SNT(\mu_1=50,\mu_2=70,\sigma_1=3,\sigma_2=5,a=50,b=\infty)$:

```
import numpy as np
from scipy.stats import norm
from sntn.dists import SNTN, tnorm

# Ground truth
nsim = 1000
mu1_seq = np.linspace(50, 60, 10)
mu2 = 70
sig1 = 3
sig2 = 5
dist1 = tnorm()
dist2 = norm(loc=mu2, scale=sig2)

for mu1 in mu1_seq:
    mu1
```

# Contributing

For testing, please set up the sntn conda environment: `conda env create -f env.yml`, and check that all unittests work as expected: `python3 -m pytest tests`. If any package changes are made, please run `conda env export > env.yml` as part of any pull request. 

## Bivariate normal approximation methods

Johnson & Katz:
https://rdrr.io/cran/weightedScores/man/approxbvncdf.html

Fast CDF:
https://github.com/david-cortes/approxcdf


