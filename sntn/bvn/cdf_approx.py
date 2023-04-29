"""
Contains the Cox method to approximate Bivariate CDF
"""

# External
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import owens_t
# Internal
from sntn.bvn.utils import Phi, phi


class bvn_cox():
    def __init__(self, monte_carlo:bool=False, nsim:int=1000, seed:int or None=None):
        """Method for obtaining the CDF of a bivariate normal using Cox's 1991 approximation method
        
        Parameters
        ----------
        monte_carlo:                Should random data be drawn to estimate P(X2 > k | X1 > h)? default=False <--> we use ~E[ Phi((rho*X1 - k)/sqrt(1-rho^2)) | X1 > h] 
        nsim:                       If MC used, how many simulated points to draw?
        seed:                       If MC used, how the draws should be seeded
        """
        # Input checks
        assert isinstance(nsim, int) and (nsim > 0), 'nsim needs to be a strictly positive int'
        assert isinstance(monte_carlo, bool), 'monte_carlo needs to be a bool'


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
    nsim:          How many draws from the truncated normal to take (default=1000)
    seed:          Will be passed into rvs method of truncnorm (default=None)
    """
    dist = truncnorm(loc=0, scale=1, a=h, b=np.infty)
    rvs = dist.rvs((nsim,)+h.shape, seed)
    orthant = Phi(-h) * np.mean(Phi((rho*rvs - k)/np.sqrt(1-rho**2)),0)
    cdf = orthant_to_cdf(orthant, h, k)
    return cdf


# --- (ii) Cox2 --- #
def cdf_cox2(h:np.ndarray, k:np.ndarray, rho:np.ndarray, rho_thresh:float=0.9, first_order:bool=False) -> np.ndarray:
    """
    Approximates the orthant probability using approach suggested by Cox (1991) and then converts to CDF. Uses a blended averages of equation 3/4, where weight is 50/50 as rho approaches 1

    Parameters
    ----------
    ...                     See cdf_cox1
    rho_thresh: 
    """
    # Process inputs
    h, k, rho = process_h_k_rho(h, k, rho)
    # If there are any rho's above 90%, we will adjust the estimate
    idx_tail = np.abs(rho) > rho_thresh
    if np.any(idx_tail):
        1
    # Convert to cdf and return
    cdf = orthant_to_cdf(orthant, h, k)
    return cdf
    

def _orthant_cox_rho(h:np.ndarray, k:np.ndarray, rho:np.ndarray, first_order:bool=False) -> np.ndarray:
    """Wrapper for _orthant_cox_rho_pos to apply both positive and negative rho's"""
    idx_rho_pos = rho >= 0
    orthant = np.zeros(h.shape)
    if np.any(idx_rho_pos):
        orthant[idx_rho_pos] = _orthant_cox_rho_pos(h[idx_rho_pos], k[idx_rho_pos], rho[idx_rho_pos], first_order)
    if np.any(~idx_rho_pos):
        orthant[~idx_rho_pos] =  Phi(-h[~idx_rho_pos]) - _orthant_cox_rho_pos(h[~idx_rho_pos], -k[~idx_rho_pos], -rho[~idx_rho_pos], first_order)
    return orthant
    

def _orthant_cox_rho_pos(h:np.ndarray, k:np.ndarray, rho:np.ndarray, first_order:bool=False) -> np.ndarray:
    """
    Calculate the cox approximation for positive values of rho
    
    Parameters
    ----------
    ...                     See cdf_cox1
    """
    # Initialize
    orthant = np.zeros(h.shape)
    # (i) At least one positive, rho positive
    idx_pos = (h > 0) | (k > 0)
    if idx_pos.any():
        h1 = np.maximum(h[idx_pos], k[idx_pos])
        k1 = np.minimum(h[idx_pos], k[idx_pos])
        orthant[idx_pos] = _orthant_cox(h1, k1, rho[idx_pos], first_order)
    # (ii) both are negative
    if np.any(~idx_pos):
        h2 = -np.maximum(-h[~idx_pos], -k[~idx_pos])
        k2 = -np.minimum(-h[~idx_pos], -k[~idx_pos])
        orthant[~idx_pos] = 1 - Phi(+h2) - Phi(+k2) + _orthant_cox(-h2, -k2, rho[~idx_pos], first_order)
    return orthant
    
def _orthant_cox(h:np.ndarray, k:np.ndarray, rho:np.ndarray, first_order:bool=False) -> np.ndarray:
    """
    Approximates the orthant probability using inverse mills estimate (see eq. 3 in paper "A Simple Approximation for Bivariate & Triviate Normal Integrals")

    L(h, k, rho) = P(X1 > h)*P(X2 > k | X1 > h)
                 ~ Phi(-h) * Phi((rho*u[h] - k)/sqrt(1-rho^2))
                 = Phi(-h) * Phi(xi(h,k,rho))
                 u[h] = E(X1 | X1 > h)
                 xi(h,k,rho) = (rho*u[h] - k)/sqrt(1-rho^2)
    
    A first order approximation of E[g(Z)] ~ g(E(z)) + 0.5*var(z)*g''(E(z)), leads to a refinement of eq. 3:

    L(h, k, rho) ~ Phi(-h) * [ Phi(xi) - 0.5*rho^2/(1-rho^2)*xi*phi(xi)*sigma^2(h)  ]
                   sigma2(h) = 1 + u[h] - u^2[h]
    
    Parameters
    ----------
    ...                     See cdf_cox1
    first_order:            Whether a first order expansion should be used (see eq. 4), default=False
    """
    # Generate approximations
    u_h = imills(h)
    xi = (rho*u_h - k)/np.sqrt(1-rho**2)
    Phi_h = Phi(-h)
    Phi_xi = Phi(xi)
    if first_order:  # eq. 4
        sigma2_h = 1 + u_h - u_h**2
        orthant = Phi_h * ( Phi_xi - 0.5*rho**2/(1-rho**2)*xi*phi(xi)*sigma2_h )
    else:  # eq. 3
        orthant = Phi_h * Phi_xi
    return orthant