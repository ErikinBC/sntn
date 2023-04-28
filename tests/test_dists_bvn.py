"""
Makes sure that the BVN class works as expected

python3 -m pytest tests/test_dists_bvn.py -s
"""

# Internal
import pytest
import numpy as np
from scipy.stats import norm, binom
# External
from sntn.dists import bvn
from parameters import seed
from sntn._bvn import valid_cdf_approach
from sntn.utilities.utils import flip_last_axis, rho_debiased

# Used for pytest
params_shape = [((1,)), ((12, )), ((4, 3)), ((3, 2, 2)),]


def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generating BVN parameters"""
    np.random.seed(seed)
    mu1 = np.random.randn(*shape)
    mu2 = np.random.randn(*shape)
    rho = 2*np.random.rand(*shape) - 1
    sigma21 = np.exp(np.random.randn(*shape)) + 1
    sigma22 = np.exp(np.random.randn(*shape)) + 1
    return mu1, sigma21, mu2, sigma22, rho


@pytest.mark.parametrize("shape", params_shape)
def test_bvn_cdf(shape:tuple, ndraw:int=100000) -> None:
    """Make sure that the scipy CDF method words"""
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    # Draw data and see how each method considers the point on the CDF
    dist = bvn(mu1, sigma21, mu2, sigma22, rho)
    x = dist.rvs(1)
    for method in valid_cdf_approach:
        dist = bvn(mu1, sigma21, mu2, sigma22, rho, cdf_approach=method)
        pval_method = dist.cdf(x)
        breakpoint()

    # Draw a large amount of data for each parameter and compare
    for kk in np.ndindex(mu1.shape):
        mu1_kk, sigma21_kk, mu2_kk, sigma22_kk, rho_kk = mu1[kk], sigma21[kk], mu2[kk], sigma22[kk], rho[kk]
        dist_kk = bvn(mu1_kk, sigma21_kk, mu2_kk, sigma22_kk, rho_kk)
        data_kk = dist_kk.rvs(ndraw)


@pytest.mark.parametrize("shape", params_shape)
def test_bvn_rvs(shape:tuple, ndraw:int=250, nsim:int=1000, tol=0.005) -> None:
    """Checks that the rvs method returns the expected empirical moments"""
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    dist = bvn(mu1, sigma21, mu2, sigma22, rho, cdf_approach='scipy')
    # Loop over nsim and store the data
    holder_rho = np.zeros((nsim,) + rho.shape)
    holder_mu = np.zeros((nsim,2,) + mu1.shape)
    for i in range(nsim):
        x = dist.rvs(ndraw, i)  # Draw data
        # Moments
        mu_hat = np.mean(x, 0)
        rho_hat = rho_debiased(np.take(x, indices=0, axis=1),np.take(x, indices=1, axis=1))
        # Store
        holder_rho[i] = rho_hat
        holder_mu[i] = mu_hat
    # Calculate z-scores 
    mu0 = np.expand_dims(np.stack([mu1,mu2],axis=0), 0)
    Sigma0 = dist.Sigma[:,[0,3]].reshape(mu1.shape+(2,))
    Sigma0 = np.expand_dims(np.sqrt(flip_last_axis(Sigma0) / ndraw), 0)
    zscore = (holder_mu - mu0) / Sigma0
    # Check that type-1 error is as expected
    alpha_check = [0.01, 0.05, 0.1]
    ndim = int(np.prod(zscore.shape))
    pval_mu = np.array([binom(n=ndim, p=alpha).cdf(np.sum(norm.cdf(zscore) < alpha)) for alpha in alpha_check])
    pval_mu = 2*np.minimum(pval_mu, 1-pval_mu)
    assert np.all(pval_mu > 0.05), 'nulls for mean difference failed'
    # Check that the difference in correlations is bounded
    err_rho = np.max(np.abs(np.mean(holder_rho, 0) - rho))
    assert err_rho < tol, f'Expected average rhos to be within {tol}: {err_rho}'

