"""
Brute force approaches for BVN integral
"""

# External
import numpy as np
from scipy.stats import multivariate_normal as mvn
# Internal
from sntn._cdf_bvn._utils import _bvn_base, mvn_pivot


# if method == 'sheppard':
#     pval = np.array([quad(self.sheppard, np.arccos(self.rho), np.pi, args=(y1,y2))[0] for y1, y2 in zip(Y1,Y2)])
#     return pval


class _bvn_scipy(_bvn_base):
    def __init__(self, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray):
        super().__init__(mu1, mu2, sigma21, sigma22, rho)


    def cdf(self, x1:np.ndarray, x2:np.ndarray, monte_carlo=None) -> np.ndarray:
        # Converts to normalized units
        h, k, rho = mvn_pivot(x1, x2, self.mu1, self.mu2, self.sigma21, self.sigma22, self.rho)
        h_shape = h.shape
        n_h_shape = len(h_shape)
        # Initialize holders
        pval = np.zeros(h.shape)
        Sigma = np.array([[1,0],[0,1]], dtype=h.dtype)
        # If h.shape == mu1.shape, then broadcasting did not affect values, and we assume we need to apply a 1:1 mapping, however if h.shape > mu1.shape, when clearly there are redundant parameters that can be called more efficiently with cdf
        if (self.n_param_shape == 0) or (self.n_param_shape==1 and sum(self.param_shape)==1):
            # parameters are floats so, cdf can be vectorized
            Sigma[[0,1],[1,0]] = self.rho
            data = np.stack([h,k])
            assert data.shape[0] == 2, 'expected first dimension to be of size 2'
            n = int(np.prod(data.shape[1:]))
            pval = mvn(cov=Sigma).cdf(data.reshape([2, n]).T)
            pval = pval.reshape(h.shape)
        elif self.n_param_shape == n_h_shape:
            # Assumes we need a 1:1 mapping
            for ii in np.ndindex(h.shape):
                Sigma[[0,1],[1,0]] = rho[ii]
                pval[ii] = mvn(cov=Sigma).cdf(np.r_[h[ii], k[ii]])
        else:
            # We should be able to recycle coordinates
            assert self.n_param_shape < n_h_shape
            # Determine which dimensions are "recycles"
            n_dim_recycle = n_h_shape - self.n_param_shape
            for ii in np.ndindex(h_shape[:n_dim_recycle]):
                # Make a recursive call...
                pval[*ii,...] = self.cdf(x1[*ii,...], x2[*ii,...])
        return pval
