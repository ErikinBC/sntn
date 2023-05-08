"""
Utilities for BVN
"""

# External libraries
import numpy as np
from scipy.stats import norm

class _bvn_base():
    def __init__(self, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray):
        # Input checks
        assert np.all((rho >= -1) & (rho <= 1)), 'rho must be b/w [-1,1]'
        # Process data
        self.mu1, self.mu2, self.sigma21, self.sigma22, self.rho = np.broadcast_arrays(mu1, mu2, sigma21, sigma22, rho)
        self.param_shape = self.mu1.shape
        self.n_param_shape = len(self.mu1.shape)


def Phi(x:np.ndarray) -> np.ndarray:
    """Returns CDF of a standard normal"""
    return norm.cdf(x)

def phi(x:np.ndarray) -> np.ndarray:
    """Returns PDF of a standard normal"""
    return norm.pdf(x)


def imills(a:np.ndarray) -> np.ndarray:
    """Returns the inverse mills ratio"""
    return norm.pdf(a)/norm.cdf(-a)


def cdf_to_orthant(cdf:np.ndarray, h:np.ndarray, k:np.ndarray):
    """The orthant and CDF probabilities have a 1:1 mapping:
    
    P(X > h, X > k) = 1 - (Phi(h) + Phi(k)) + MVN(X <= h, X<= k)
    """
    L_hk = 1 - (norm.cdf(h) + norm.cdf(k)) + cdf
    return L_hk


def orthant_to_cdf(orthant:np.ndarray, h:np.ndarray, k:np.ndarray):
    """The orthant and CDF probabilities have a 1:1 mapping:
    
    P(X <= h, X<= k) = P(X > h, X > k) - 1 + (Phi(h) + Phi(k))
    """
    P_hk = orthant - 1 + norm.cdf(h) + norm.cdf(k)
    return P_hk


def mvn_pivot(x1:np.ndarray, x2:np.ndarray, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray) -> tuple:
    """
    For any BVN([mu1,mu2],[sigma21,root(sigma21*sigma22)*rho,.,sigma22]).cdf(x1,x2), the CDF can be calculate for a standard normal BVN: BVN([0,0],[1,rho,rho,1]).cdf(h,k), where
    
    h:              (x1 - mu1) / sqrt(sigma21)
    k:              (x2 - mu2) / sqrt(sigma22)
    """
    x1, x2, mu1, mu2, sigma21, sigma22, rho = np.broadcast_arrays(x1, x2, mu1, mu2, sigma21, sigma22, rho)
    h = (x1 - mu1) / np.sqrt(sigma21)
    k = (x2 - mu2) / np.sqrt(sigma22)
    return h, k, rho

