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



###########################
# ---  APPROXIMATIONS --- #


import pandas as pd
from scipy.stats import multivariate_normal as mvn
mu = np.array([+1,-1.5])
rho = -0.95
sigma21 = 3
sigma22 = 2
sigma1, sigma2 = np.sqrt(sigma21), np.sqrt(sigma22)
sigma_rho = np.sqrt(sigma21) * np.sqrt(sigma22) * rho
Sigma = np.array([[sigma21,sigma_rho],[sigma_rho,sigma22]])
np.random.seed(1)
n = 10
X0 = np.random.randn(n, 2)
h, k, rho = mvn_pivot(X0[:,0], X0[:,1], mu[0], mu[1], sigma21, sigma22, rho)
np.c_[h, k]
# mvn(mu, Sigma).cdf(X0)[(h < 0) & (k < 0)]
pd.DataFrame({'s':mvn(mu, Sigma).cdf(X0),'c1':cdf_cox2(h, k, rho, False),'c2':cdf_cox2(h, k, rho, True)})



# --- (i) Cox3 --- #
def cdf_cox3(h:np.ndarray, k:np.ndarray, rho:np.ndarray) -> np.ndarray:
    """
    Approximates the orthant probability using second order approximation (see eq. 4 in paper "A Simple Approximation for Bivariate & Triviate Normal Integrals")

    L(h, k, rho) = P(X1 > h)*P(X2 > k | X1 > h)
                 ~ Phi(-h) * Phi((rho*u[h] - k)/sqrt(1-rho^2))
                 = Phi(-h) * Phi(xi(h,k,rho))
                 u[h] = E(X1 | X1 > h)
                 xi(h,k,rho) = (rho*u[h] - k)/sqrt(1-rho^2)
    
    Parameters
    ----------
    See cdf_cox1
    """
    return None



# --- (iv) Owen --- #
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


# --- (v) Lin --- #

# --- (vi) Dresner --- #


#######################
# ---  QUADRATURE --- #

# for loops: https://docs.scipy.org/doc/scipy/tutorial/integrate.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html