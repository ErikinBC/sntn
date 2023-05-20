## `sntn`: Sum of Normal and Truncated Normal package

This repo is for the `sntn` package, which implements a scipy-like class for doing inference on a sum of a normal and a trunctated normal (SNTN). See the arXiv paper for more details on how this distribution can be used for data carving (a type of post-selection inference). The SNTN distribution has eight parameters:

$$
z \sim 
$$

# Installation

see pypi.

To check that the package compiled properly, please run `python3 -m sntn`.

# main classes

1. `dists.nts(mu1, tau21, mu2, tau22, a, b, c1, c2)`: main class for the doing inference on a SNTN distribution, with the usual scipy-like methods: `cdf`, `pdf`, `ppf`, and `rvs` as well as a `conf_int` method for generating exact confidence intervals.
2. `dists.tnorm(mu, sigma2, a, b)`: Wrapper for key methods of a truncated normal from the `scipy.stats.truncnorm` class along with a `conf_int` method to generate exact confidence intervals for a truncated normal distribution (which can be used by a 100%-screening approach for PoSI) and matches the intervals that will get produced by the [selectiveInference package](https://cran.r-project.org/web/packages/selectiveInference/). Note that this class accepts the lower/upper bounds as is, and does not require them to be transformed in advance (as scipy does).
3. `dists.bvn(mu1, sigma21, mu2, sigma22, rho)`:  Custom bivariate normal (BVN) distribution with `cdf` and `rvs` methods. Uses can pass `cdf_approach={scipy, cox1, cox2, owen, drezner1, drezner2}` as a kwarg, with the defaul set to owen (which uses the Owens-T, and is very fast, but can be numerically instable if $|\rho| \approx 1$).
4. `posi.lasso(frac_split=0.0)`: When `frac_split=0.0`, then this is equivalent to using 100%-screening approach for PoSI.
5. `posi.marginal_screen(frac_split=0.0)`: 
6. `trialML.two_stage`: 

# Folder structure of this repo

* sntn: main package folder
* tests: unittesting folder
* examples: Jupyter notebook showing how to use the sntn package
* simulations: research work, used in arXiv paper
* figures: any figures generated by exploratory work/unittesting go here

# Toy examples

## (i) Filtering regime

Suppose an industrial process combines two parts together, but the first part is only kept if it's length exceeds a certain value. According to existing guidelines that first piece should be N(50,3) and if only kepy if >=50, and the second piece should be N(70,5). The code block below tests the power of the `SNTN` class to detect whether this distribution actually conforms to our null distribution: $SNT(\mu_1=50,\mu_2=70,\sigma_1=3,\sigma_2=5,a=50,b=\infty)$:

```
import numpy as np
from sntn.dists import sntn

....
```

### (ii) Data carving


## (iii) Conditional null hypothesis testing


# unittesting

1. test_conf_inf_solver: Makes sure that root-finding works for the Gaussian and binomial distribution
2. test_utils: Checks output from specific utility functions
3. test_grad: Makes sure that the log(f(a)+f(b)) ~ _log_gauss_approx(f(a),f(b)) works to a specific tolerance
4. test_dists_tnorm: Makes sure that the custom tnorm dist works as expected
5. test_dists_sntn: Makes sure that the main sntn dist works as expected


# Contributing

For testing, please set up the sntn conda environment: `conda env create -f env.yml`, and check that all unittests work as expected: `python3 -m pytest tests`. If any package changes are made, please run `conda env export > env.yml` as part of any pull request. 



