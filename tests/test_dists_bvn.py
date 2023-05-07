"""
Makes sure that the BVN class works as expected

python3 -m pytest tests/test_dists_bvn.py -s
"""

# Internal
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
# External
from sntn.dists import bvn
from parameters import seed
from sntn._bvn import valid_cdf_approach
from sntn.utilities.utils import flip_last_axis, rho_debiased, array_to_dataframe

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
def test_bvn_cdf(shape:tuple, ndraw:int=10, nsim:int=100000, tol:float=0.03) -> None:
    """Checks different BVN cdf methods work, with errors indexed to the scipy approach"""
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    # Draw data and see how each method considers the point on the CDF
    dist = bvn(mu1, sigma21, mu2, sigma22, rho)
    x = dist.rvs(ndraw)
    holder_pval = []
    for approach in valid_cdf_approach:
        print(f'--- Running approach {approach} ---')
        dist = bvn(mu1, sigma21, mu2, sigma22, rho, cdf_approach=approach, rho_max_w=0.25)
        pval_method = dist.cdf(x)
        tmp_df = array_to_dataframe(pval_method).melt(ignore_index=False).rename_axis('x').reset_index().assign(approach=approach)
        holder_pval.append(tmp_df)
    res_approach = pd.concat(holder_pval).reset_index(drop=True)
    
    # Draw a large amount of data for each parameter and compare
    holder_sim = []
    for kk in np.ndindex(mu1.shape):
        mu1_kk, sigma21_kk, mu2_kk, sigma22_kk, rho_kk = mu1[kk], sigma21[kk], mu2[kk], sigma22[kk], rho[kk]
        dist_kk = bvn(mu1_kk, sigma21_kk, mu2_kk, sigma22_kk, rho_kk)
        data_kk = np.squeeze(dist_kk.rvs(nsim))
        # Extract x
        x_kk = x[...,*kk,:]
        assert x_kk.shape == (ndraw, 2), 'Did not extract as expected'
        pval_kk = np.zeros(ndraw)
        for i in range(ndraw):
            # Calculate CDF value
            pval_kk[i] = np.mean((data_kk[:,0] <= x_kk[i,0]) & (data_kk[:,1] <= x_kk[i,1]))
        tmp_kk = pd.DataFrame({'value':pval_kk})
        for j, k in enumerate(kk):
            tmp_kk.insert(0,f'd{j+1}',k)
        holder_sim.append(tmp_kk)
    # Combine
    res_rvs = pd.concat(holder_sim).rename_axis('x').reset_index().assign(approach='rvs')

    # Merge and compare
    cn_idx = list(np.setdiff1d(res_rvs.columns,['value','approach']))
    res_wide = pd.concat(objs=[res_approach, res_rvs],axis=0).pivot(index=cn_idx,columns='approach',values='value')
    cn_gt = 'scipy'
    res_long = res_wide.melt(cn_gt,ignore_index=False).assign(aerr=lambda x: (x['value']-x[cn_gt]).abs())
    max_err = res_long['aerr'].max()
    assert max_err < tol, f'Woops! {max_err} is greater than {tol}'
    


@pytest.mark.parametrize("shape", params_shape)
def test_bvn_rvs(shape:tuple, ndraw:int=250, nsim:int=1000, tol=0.005) -> None:
    """Checks that the rvs method returns the expected empirical moments"""
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    dist = bvn(mu1, sigma21, mu2, sigma22, rho, cdf_approach='scipy')
    # Loop over nsim and store the data
    holder_rho = np.zeros((nsim,) + rho.shape)
    holder_mu = np.zeros((nsim,) + mu1.shape + (2,))
    for i in range(nsim):
        x = dist.rvs(ndraw, i)  # Draw data
        # Moments
        mu_hat = np.mean(x, 0)
        rho_hat = rho_debiased(np.take(x, indices=0, axis=-1),np.take(x, indices=1, axis=-1))
        # Store
        holder_rho[i] = rho_hat
        holder_mu[i] = mu_hat
    # Calculate z-scores 
    mu0 = np.expand_dims(np.stack([mu1,mu2],axis=-1), 0)
    Sigma0 = dist.Sigma[:,[0,3]].reshape(mu1.shape+(2,))
    Sigma0 = np.expand_dims(np.sqrt(Sigma0 / ndraw), 0)
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

