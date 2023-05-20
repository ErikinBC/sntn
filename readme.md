## `sntn`: The sum of a normal and truncated normal distribution package


This repo is for the `sntn` package, which implements a scipy-like class for doing inference on a sum of a normal and a trunctated normal (SNTN) distribution (see [Kim (2006)](https://www.kss.or.kr/jounalDown.php?IDX=831) and [Arnold (1993)](https://link.springer.com/article/10.1007/BF02294652)). 

The SNTN distribution can be used in a variety of situations including [data carving](https://arxiv.org/abs/) (a type of post-selection inference), situations where a filtering process is applied, as well as [two-stage hypothesis testing](http://www.erikdrysdale.com/regression_trial/#1-two-stage-testing-approach). The [arXiv paper](https://arxiv.org/abs/), "A parametric distribution for exact post-selection inference with data carving" makes extensive use of this package. Please see the (notebook)[examples/data_carving.ipynb] in the examples folder for a more thorough walk through of how these methods can be used for the post-selection inference lasso and marginal screening algorithms.

Formally, if $X_1 \sim N(\mu_1, \tau_1^2)$ and $X_2 \sim \text{TN}(\mu_2, \tau_2^2, a, b)$, then $Z = c_1 X_1 + c_2 X_2$, $c_i \in \mathbb{R}$, is said to follow an SNTN distribution denoted as either $\text{SNTN}(\mu_1, \tau_1^2, \mu_2, \tau_2^2, a, b, c_1, c_2)\overset{d}{=}\text{SNTN}(\theta_1, \sigma_1^2, \theta_2, \sigma_2^2, \omega, \delta)$ where $\theta_1=\sum_{i=1}^2c_i\mu_i$, $\sigma_1^2=\sum_{i=1}^2c_i^2\tau_i^2$, $\theta_2=\mu_2$, $\sigma_2^2=\tau_2^2$, $m_j(x)=(x-\theta_x)/\sigma_x$, $\omega=m_2(a)=(a-\theta_2)/\sigma_2$, and $\delta=m_2(b)=(b-\theta_2)/\sigma_2$.

Inference for the SNTN distributions ends up being computationally scalable since the cdf ($f$) has a closed form solution, and the CDF ($F$) can be calculated using the CDF of two bivariate normal distributions with correlation $\rho$.

$$
\begin{align*}
f_{\theta,\sigma^2}^{\omega,\delta}(z) &= \frac{\phi(m_1(z))[\Phi(\gamma\delta -\lambda m_1(z)) - \Phi(\gamma\omega -\lambda m_1(z))]}{\sigma_1[\Phi(\delta)-\Phi(\omega)]} \\
F_{\theta,\sigma^2}^{\omega,\delta}(z) &= \frac{B_\rho(m_1(z),\delta) - B_\rho(m_1(z),\omega)}{\Phi(\delta)-\Phi(\omega)}  \\
\rho&=c_2\sigma_2/\sigma_1 \\
\lambda &= \rho/\sqrt{1-\rho^2} \\
\gamma &= \lambda/\rho
\end{align*}
$$

<br>

# Installation

see pypi.

To check that the package compiled properly, please run `python3 -m sntn`.

<br>

# main classes

There are six classes from this package that are likely to be used by practioneers. Functions or classes that start with an underscore "_" are meant to be internally used, but some of the their optional arguments may be of interest.

1. `dists.nts(mu1, tau21, mu2, tau22, a, b, c1=1, c2=1)`: main class for the doing inference on a SNTN distribution, with the usual scipy-like methods: `cdf`, `pdf`, `ppf`, and `rvs` as well as a `conf_int` method for generating exact confidence intervals (default is to assume that the X's are equally weighted c1==c2==1). Optional keyword arguments include `fix_mu:bool` which forces mu1==mu2, which is useful for generating confidence intervals or quantiles when the nulll hypothesis is that both $X_1$ and $X_2$ have the same underlying mean, as well as any other named parameter which can go into `dists._bvn` (discussed below). 
2. `dists.tnorm(mu, sigma2, a, b)`: Wrapper for key methods of a truncated normal from the `scipy.stats.truncnorm` class along with a `conf_int` method to generate exact confidence intervals for a truncated normal distribution (which can be used by a 100%-screening approach for PoSI) and matches the intervals that will get produced by the [selectiveInference package](https://cran.r-project.org/web/packages/selectiveInference/). Note that this class accepts the lower/upper bounds as is, and does not require them to be transformed in advance (as scipy does). Both `nts` and `tnorm` use `_solvers.conf_inf_solver._conf_int` to find confidence intervals and kwargs can be passed into it.
3. `dists.bvn(mu1, sigma21, mu2, sigma22, rho)`:  Custom bivariate normal (BVN) distribution with `cdf` and `rvs` methods. Uses can pass `cdf_approach={scipy, cox1, cox2, owen, drezner1, drezner2}` as a kwarg, with the default set to owen (which uses the Owens-T, and is very fast, but can be numerically instable if $|\rho| \approx 1$). Each cdf_approach has its own kwargs which can be passed in during construction (e.g. `_cdf_bvn._approx._bvn_cox`).
4. `posi.lasso(lam, y, x, frac_split=0.5, seed=None)`: Carries out post-selection with the Lasso for a fixed value of $\lambda$, where `frac_split` is the proportion of samples that will be used by the inference half (i.e. the part only used for inference). It is recommended to give a `seed` for reproducability (see `_split._split_yx` for other kwargs that can be passed). The `run_inference(alpha, null_beta)` is the main method to be called and will store the difference statistical tests in DataFrames (see description for attribute names).
5. `posi.marginal_screen(k,...)`: Similar to the Lasso, except that the algorithm which selects coefficients is a marginal screening one.
6. `trialML.two_stage`: Carries out a statistical test for a two-stage regression hypothesis testing scenario (still under construction).


# Examples

## (i) Filtering regime

In some manufacturing processes, one of the components may go through a quality control procedure that removes items above or below a certain threshold. For example, this question was posed in a old issue of [Technometrics](https://www.jstor.org/stable/1266101?seq=1):

> An item which we make has, among others, two parts which are assembled additively with regard to length. The lengths of both parts are normally distributed but, before assembly, one of the parts is subjected to an inspection which removes all individuals below a specified length. As an example, suppose that X comes from a normal distribution with a mean 100 and a standard deviation of 6, and Y comes from a normal distribution with a mean of 50 and a standard deviation of 3, but with the restriction that Y > 44. How can I find the chance that X + Y is equal to or less than a given value?

Subsequent answers focused on value of $P(X+Y<138) \approx 0.03276$. We can confirm this by using the `nts` class.

```
import numpy as np
from sntn.dists import nts

mu1, tau21 = 100, 6**2
mu2, tau22 = 50, 3**2
a, b = 44, np.inf
w = 138
dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
cdf_1964 = dist_1964.cdf(w)[0]
print(f'Probability that Z<138={cdf_1964*100:0.3f}%')
```
 Probability that Z<138=3.276%


### (ii) Data carving

Please see the (notebook)[examples/data_carving.ipynb] in the examples folder for a more thorough walk through of how these methods can be used for the post-selection inference lasso and marginal screening algorithms.

Consider a high-dimensional regression problem with many a dozen small covariates with a true signal and many more that are just noise covariates. We'll compare how many true and false positives are detected by 100% screening (only Lasso), 50/50% data splitting, versus 90/10% data carving. 

```
seed = 2
alpha = 0.05
bhat_null = 0
n, p, s = 100, 150, 12
beta0 = np.zeros(p)
b = 0.35
beta0[:s] = b
np.random.seed(seed)
x = np.random.randn(n,p)
g = x.dot(beta0)
u = np.random.randn(n)
y = g + u
lammax = np.max(np.abs(x.T.dot(y)/n))
lam = lammax * 0.5
# Lasso only
inf_posi = lasso(lam, y, x, frac_split=0)
inf_posi.run_inference(alpha, null_beta, sigma2=1.0, run_carve=False, run_split=False, run_ci=False)
inf_posi = inf_posi.res_screen.query('pval < @alpha')
idx_tp_lasso = inf_posi['cidx'] < s
tp_lasso, fp_lasso = np.sum(idx_tp_lasso), np.sum(~idx_tp_lasso)
# 50/50 split
inf_split = lasso(lam, y, x, frac_split=0.5, seed=seed)
inf_split.run_inference(alpha, null_beta, sigma2=1.0, run_carve=False, run_ci=False)
inf_split = inf_split.res_screen.query('pval < @alpha')
idx_tp_split = inf_split['cidx'] < s
tp_split, fp_split = np.sum(idx_tp_split), np.sum(~idx_tp_split)
print(f'Lasso (TP={tp_lasso}, FP={fp_lasso})\nSplit (TP={tp_split}, FP={fp_split})')
# 90/10 carve
inf_carve = lasso(lam, y, x, frac_split=0.1, seed=seed)
inf_carve.run_inference(alpha, null_beta, sigma2=1.0, run_ci=False)
inf_carve = inf_carve.res_carve.query('pval < @alpha')
idx_tp_carve = inf_carve['cidx'] < s
tp_carve, fp_carve = np.sum(idx_tp_carve), np.sum(~idx_tp_carve)
print(f'Lasso (TP={tp_lasso}, FP={fp_lasso})\nSplit (TP={tp_split}, FP={fp_split})\nCarve (TP={tp_carve}, FP={fp_carve})')
```
 Lasso (TP=1, FP=0)

 Split (TP=1, FP=1)
 
 Carve (TP=2, FP=0)
 






<br>

# Folder structure of this repo

* sntn: main package folder
* tests: unittesting folder
* examples: Jupyter notebook showing how to use the sntn package
* simulations: research work, used in arXiv paper
* figures: any figures generated by exploratory work/unittesting go here
* R: Checks that the selectiveInference package in R aligns with the PoSI truncated-normal estimates

<br>


# Contributing

For testing, please set up the sntn conda environment: `conda env create -f env.yml`, and check that all unittests work as expected: `python3 -m pytest tests`. If any package changes are made, please run `conda env export > env.yml` as part of any pull request. 

## unittests

1. test_conf_inf_solver: Makes sure that root-finding works for the Gaussian and binomial distribution
2. test_utils: Checks output from specific utility functions
3. test_grad: Makes sure that the log(f(a)+f(b)) ~ _log_gauss_approx(f(a),f(b)) works to a specific tolerance
4. test_dists_tnorm: Makes sure that the `_tnorm` dist works as expected
5. test_dists_nts: Makes sure that the main `_nts` dist works as expected
6. test_dist_bvn: Makes sure that the main `_bvn` dist works as expected 
7. test_posi: Checks that the marginal screening wrapper `_posi_marginal_screen` has the expected statistical performance.
8. test_xvar_carve: Checks that the `nts` class obtains the expected statistical performance for data carving with the sample mean


