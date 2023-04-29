"""
Collection of bivariate normal CDF solvers
"""

# External
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import owens_t
from scipy.integrate import quad
# Internal
from sntn.dists import tnorm

#####################
# --- UTILITIES --- #

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
    h = (x1 - mu1) / np.sqrt(sigma21)
    k = (x2 - mu2) / np.sqrt(sigma22)
    return h, k


###########################
# ---  APPROXIMATIONS --- #

# --- (i) Cox1 --- #
def cdf_cox1(h:np.ndarray, k:np.ndarray, rho:np.ndarray, nsim:int=1000, seed:int or None=None) -> np.ndarray:
    """
    Implements a Monte Carlo approach of estimating the orthant probability:

    L(h, k, rho) = P(X1 > h)*P(X2 > k | X1 > h)
                 = Phi(-h) * E[ Phi((rho*X1 - k)/sqrt(1-rho^2)) | X1 > h] 
    
    Where X1 > h is simply a truncated Gaussain: TN(0,1,h,infty)

    Parameters
    ----------
    h:             A (d1,d2,...,dk) array of normalized X1's
    k:             A (d1,d2,...,dk) array of normalized X1's
    rho:           A (d1,d2,...,dk) array of correlation coefficients
    """
    dist = truncnorm(loc=0, scale=1, a=h, b=np.infty)
    rvs = dist.rvs((nsim,)+h.shape, seed)
    orthant = norm.cdf(-h) * np.mean(norm.cdf((rho*rvs - k)/np.sqrt(1-rho**2)),0)
    cdf = orthant_to_cdf(orthant, h, k)
    return cdf


import pandas as pd
from scipy.stats import multivariate_normal as mvn
mu = np.array([1,-1])
rho = -0.1
sigma21 = 3
sigma22 = 2
sigma1, sigma2 = np.sqrt(sigma21), np.sqrt(sigma22)
sigma_rho = np.sqrt(sigma21) * np.sqrt(sigma22) * rho
Sigma = np.array([[sigma21,sigma_rho],[sigma_rho,sigma22]])
np.random.seed(1)
n = 10
X0 = np.random.randn(n, 2)
h, k = mvn_pivot(X0[:,0], X0[:,1], mu[0], mu[1], sigma21, sigma22, rho)
pd.DataFrame({'s':mvn(mu, Sigma).cdf(X0),'c':cox1(h, k, rho, nsim=10000, seed=1)})




# --- (ii) Cox2 --- #

# --- (iii) Owen --- #
def cdf_owen(h:np.ndarray, k:np.ndarray, rho:np.ndarray) -> np.ndarray:
    """Owen's (1956) method"""
    # Calculate delta_hk (either zero or one)
    delta_hk = np.where(h*k>=0, 0, 1)  #delta_hk = np.where(h*k>=0, 0, np.where(h + k >= 0, 0, 1))
    # Calculate constants for Owen's T
    den_rho = np.sqrt(1 - rho**2)
    q1 = (k / h - rho) / den_rho
    q2 = (h / k - rho) / den_rho
    # These are the computation bottleneck
    t1 = owens_t(h, q1)
    t2 = owens_t(k, q2)
    cdf = 0.5 * (norm.cdf(h) + norm.cdf(k) - delta_hk) - t1 - t2 
    return cdf


x = np.random.randn(100,2)
from timeit import timeit
timeit('owens_t(x[:,0], x[:,1])',globals=globals(), number=100000)



# --- (iv) Lin --- #

# --- (v) Dresner --- #


#######################
# ---  QUADRATURE --- #