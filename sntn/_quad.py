"""
Contains the custom functions needed to estimate the CDF/Quantile of the NTS distribution using custom integral method
"""

# Load modules
import numpy as np
from scipy.stats import norm
from . import type_farray

def _integrand_X12(x1: type_farray, x2: type_farray, rho: type_farray) -> float | np.ndarray:
    """See bvn_cdf_diff"""
    return norm.cdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)


def bvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray, n_points: int=1001) -> float | np.ndarray:
    """
    Calculates the difference in the CDF between two bivariate normals with a shared x1 and rho value:

    BVN(rho).cdf(x1, x2a) - BVN(rho).cdf(x1, x2b)

    Background
    ==========
    BVN(rho).cdf(x1, x2) = int_{-infty}^{x2} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    Since X1 | X2=z ~ N(rho*z, 1-rho^2)

    So the difference in the integrals is simply:
    int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz
    """
    # Broadcast...
    breakpoint()
    # ...
    d_points = (x2a - x2b) / (n_points - 1)
    points = np.linspace(x2b, x2a, num=n_points)
    y = _integrand_X12(x1=x1, x2=points, rho=rho)
    y = np.squeeze(y)
    int_f = np.trapz(y, dx=d_points)
    return int_f

def _integral110(z, a, b):
        fb = np.sqrt(1 + b**2)
        res = (1/fb) * norm.pdf(a / fb) * norm.cdf(z*fb + a*b/fb)
        return res

def dbvn_cdf_diff(x1, x2a, x2b, rho) -> float | np.ndarray:
    """
    Calculates the derivative of the integral:

    I(x1; x2a, x2b, rho) = int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    w.r.t x1:
    
    dI/dx1 = 1/sqrt(1-rho^2) int_{x2b}^{x2a} phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz

    This nicely has a closed form solution (see https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions and Owen 1980)
    """

    frho = np.sqrt(1-rho**2)
    a = x1 / frho
    b = -rho / frho
    val = (_integral110(x2a, a, b) - _integral110(x2b, a, b)) / frho
    return val