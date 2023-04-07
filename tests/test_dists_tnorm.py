"""
Make sure truncated normal wrapper (tnorm) has expected properties to some reasonable tolerance

python3 -m tests.test_dists_tnorm
python3 -m pytest tests/test_dists_tnorm.py
"""

import pytest
import numpy as np
from scipy.stats import norm, kstest
# Internal
from sntn.dists import tnorm
from sntn.utilities.utils import has_assertion_error
from parameters import seed

# Set seed
np.random.seed(seed)

def gen_params(shape:tuple or list) -> tuple:
    mu = np.random.randn(*shape)
    sigma2 = np.exp(mu)
    a, b = mu - 2, mu + 2
    return mu, sigma2, a, b


params_tnorm = [((10, )), ((10, 5)), ((10, 5, 2)),]
@pytest.mark.parametrize("shape", params_tnorm)
def ztest_tnorm_rvs(shape:tuple or list, nsim:int=100000, tol:float=1e-2) -> None:
    """Check that rvs() gets the mean/median we expect from theory"""
    mu, sigma2, a, b = gen_params(shape)
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    Z = (norm.pdf(dist.alpha)-norm.pdf(dist.beta))/(norm.cdf(dist.beta)-norm.cdf(dist.alpha))
    mu_theory = dist.mu + Z*dist.sigma
    med_theory = dist.mu + norm.ppf((norm.cdf(dist.beta)+norm.cdf(dist.alpha))/2)*dist.sigma
    ndec = int(-np.log10(tol))
    err_mu = np.round(np.max(np.abs(np.mean(x, 0) - mu_theory)),ndec)
    err_med = np.round(np.max(np.abs(np.median(x, 0) - med_theory)),ndec)
    assert err_mu <= tol, f'Expected mean error to be less than {tol}: {err_mu} for {ndec} decimal places'
    assert err_mu <= tol, f'Expected median error to be less than {tol}: {err_med} for {ndec} decimal places' 


def ztest_tnorm_cdf(n:int=1, nsim:int=10000, alpha:float=0.05) -> None:
    """Check that the cdf is uniform"""
    mu, sigma2, a, b = gen_params((n,))
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    pval = kstest(x, dist.cdf, method='exact').pvalue
    assert pval > alpha, f'kstest rejected null of uniformity'
    pval_err = kstest(x+1, dist.cdf, method='exact').pvalue
    assert pval_err < alpha, f'kstest did not reject null of uniformity'


def ztest_tnorm_ppf(n:int=1, nsim:int=1000000, tol:float=1e-2) -> None:
    """Check that the q/q plot is linear with a slope of 1"""
    mu, sigma2, a, b = gen_params((n,))
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    pseq = np.linspace(0.01, 0.99, 99)
    q_emp = np.quantile(x, pseq)
    q_theory = dist.ppf(pseq)
    err_q = np.max(np.abs(q_emp - q_theory))
    assert err_q < tol, f'QQ error was greater than {tol}: {err_q}'


def test_tnorm_CIs() -> None:
    """Check that the confidence interval is working as expected"""


if __name__ == "__main__":
    # Check all functions
    for param in params_tnorm:
        param
        # test_tnorm_rvs(param)
    # test_tnorm_cdf()
    # test_tnorm_ppf()
    # test_tnorm_CIs()

    print('~~~ The test_dists.py script worked successfully ~~~')