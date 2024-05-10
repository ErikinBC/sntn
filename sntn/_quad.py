"""
Contains the custom functions needed to estimate the CDF/Quantile of the NTS distribution using custom integral method
"""

# Load modules
import numpy as np
from scipy.stats import norm


def _integrand_X12(x1, x2, rho) -> float | np.ndarray:
    """See bvn_cdf_diff"""
    return norm.cdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)


def bvn_cdf_diff(x1, x2a, x2b, rho, n_points: int=1001, bound: int=10) -> float | np.ndarray:
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
    d_points = 1 / (n_points - 1)
    ub = np.where(x2a == +np.infty, +bound, x2a)
    lb = np.where(x2b == -np.infty, -bound, x2b)
    points = np.linspace(lb, ub, num=n_points)
    y = _integrand_X12(x1=x1, x2=points, rho=rho)
    # y = np.squeeze(y)
    int_f = np.trapz(y, dx=d_points)
    return int_f


def dbvn_cdf_diff(x1, x2a, x2b, rho) -> float | np.ndarray:
    """
    Calculates the derivative of the integral:

    I(x1; x2a, x2b, rho) = int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    w.r.t x1:
    
    dI/dx1 = 1/sqrt(1-rho^2) int_{x2b}^{x2a} phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz

    This nicely has a closed form solution (see Owen 1980)
    """
    def integral110(x, a, b):
        fb = np.sqrt(1 + b**2)
        res = (1/fb) * norm.pdf(a / fb) * norm.cdf(x*fb + a / fb)
        return res

    frho = np.sqrt(1-rho**2)
    a = x1 / frho
    b = -rho / frho
    val = (integral110(x2a, a, b) - integral110(x2b, a, b)) / frho
    return val