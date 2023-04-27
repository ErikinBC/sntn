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

# Used for pytest
params_shape = [((1,)), ((12, )), ((4, 3)), ((3, 2, 2)),]

# from scipy.linalg import cholesky
# sigma21 = 1.5
# sigma22 = 2.2
# rho = 0.5
# sigma1 = np.sqrt(sigma21)
# sigma2 = np.sqrt(sigma22)
# Sigma = np.array([[sigma21, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma22]])
# A = cholesky(Sigma)
# X = A.dot(np.random.randn(2, 1000)).T
# X.mean(0)
# np.corrcoef(X[:,0], X[:,1])[0,1]

# A = np.random.randn(12, 2, 2)
# B = np.random.randn(2, 1000, 12)
# # result = np.einsum('ijk,jlk->ilk', A, B)
# Bt = np.transpose(B, (2, 0, 1))
# result = np.einsum('ijk,ikl->ijl', A, Bt)
# result2 = np.stack([A[j,:,:].dot(B[:,:,j]) for j in range(12)],0)
# A[3,:,:].dot(B[:,:,3]).T
# result[3].T

# from timeit import timeit
# timeit("np.einsum('ijk,ikl->ijl', A, Bt)", globals=globals(), number=10000)
# timeit("np.stack([A[j,:,:].dot(B[:,:,j]) for j in range(12)],0)", globals=globals(), number=10000)





def rho_debiased(x:np.ndarray, y:np.ndarray, method:str='fisher'):
    """
    Return a debiased version of the emprical correlation coefficient
    
    Parameters
    ----------
    x:              A (n,i,j,k,...) dim array
    y:              Matches dim of x

    Returns
    -------
    An (i,j,k,...) array of correlation coefficients
    """
    valid_methods = ['pearson', 'fisher', 'olkin']
    assert method in valid_methods, f'method must be one of {valid_methods}'
    n = x.shape[0]
    breakpoint()
    rho = np.corrcoef(x, y, rowvar=False)
    if method == 'pearson':
        return rho
    if method == 'fisher':
        return rho * (1 + (1-rho**2)/(2*n))
    if method == 'olkin':
        return rho * (1 + (1-rho**2)/(2*(n-3)))


def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generating BVN parameters"""
    np.random.seed(seed)
    mu1 = np.random.randn(*shape)
    mu2 = np.random.randn(*shape)
    rho = 2*np.random.rand(*shape) - 1
    sigma21 = np.exp(np.random.randn(*shape)) + 1
    sigma22 = np.exp(np.random.randn(*shape)) + 1
    return mu1, sigma21, mu2, sigma22, rho

def test_bvn_rvs(ndraw:int=250, nsim:int=1000) -> None:
    """Checks that the rvs method returns the expected empirical moments"""
    shape = params_shape[1]
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    dist = bvn(mu1, sigma21, mu2, sigma22, rho, cdf_approach='scipy')
    # Loop over nsim and store the data
    nd = int(np.prod(shape))
    holder_rho = np.zeros([nsim, nd])
    holder_mu = np.zeros([nsim, nd, 2])
    for i in range(nsim):
        x = dist.rvs(ndraw, i)  # Draw data
        # Moments
        mu_hat = np.mean(x, 0).T
        rho_debiased(x[:,0,:],x[:,1,:])
        rho_hat = np.array([np.corrcoef(x[:,0,j],x[:,1,j])[0,1] for j in range(shape[0])])
        
        # Store
        holder_rho[i] = rho_hat
        holder_mu[i] = mu_hat
    # Calculate z-scores 
    zscore = (holder_mu - np.expand_dims(np.c_[mu1,mu2], 0)) / np.expand_dims(np.sqrt(dist.Sigma[:,[0,3]] / ndraw), 0)
    # Check that type-1 error is as expected
    alpha_check = [0.01, 0.05, 0.1]
    ndim = int(np.prod(zscore.shape))
    pval_mu = np.array([binom(n=ndim, p=alpha).cdf(np.sum(norm.cdf(zscore) < alpha)) for alpha in alpha_check])
    pval_mu = 2*np.minimum(pval_mu, 1-pval_mu)
    assert np.all(pval_mu > 0.05), 'nulls for mean difference failed'
    

    # Calculate the z-score for the means...
    zerr_mu = (mu_hat.T-np.c_[mu1,mu2]) / np.sqrt(dist.Sigma[:,[0,3]] / ndraw)
    holder_zscore = np.zeros([nsim, nd, 2])    
    breakpoint()
    # Check empirical mean
    
    
    err_mu = (mu_hat - np.c_[mu1,mu2].T).T
    err_mu / np.sqrt(dist.Sigma[:,[0,3]]/ndraw)
    err_mu
    
    
    np.abs()


    breakpoint()
    # EMPIRICAL CORRELATION SHOULD ALIGN WITH EXPECTED VALUE...


def test_bvn_scipy() -> None:
    """Make sure that the scipy CDF method words..."""
    None