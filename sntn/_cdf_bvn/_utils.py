"""
Utilities for BVN
"""

# External libraries
import numpy as np
from scipy.stats import norm


def Phi(x:np.ndarray) -> np.ndarray:
    """Returns CDF of a standard normal"""
    return norm.cdf(x)

def phi(x:np.ndarray) -> np.ndarray:
    """Returns PDF of a standard normal"""
    return norm.pdf(x)


def sheppard(theta:np.ndarray, h:np.ndarray, k:np.ndarray) -> np.ndarray:
    """Returns the function value for doing a BVN integral"""
    f = (1/(2*np.pi))*np.exp(-0.5*(h**2+k**2-2*h*k*np.cos(theta))/(np.sin(theta)**2))
    return f

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


def mvn_pivot(x1:np.ndarray, x2:np.ndarray, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray) -> tuple:
    """
    For any BVN([mu1,mu2],[sigma21,root(sigma21*sigma22)*rho,.,sigma22]).cdf(x1,x2), the CDF can be calculate for a standard normal BVN: BVN([0,0],[1,rho,rho,1]).cdf(h,k), where
    
    h:              (x1 - mu1) / sqrt(sigma21)
    k:              (x2 - mu2) / sqrt(sigma22)
    """
    h = (x1 - mu1) / np.sqrt(sigma21)
    k = (x2 - mu2) / np.sqrt(sigma22)
    return h, k

