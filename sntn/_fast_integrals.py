"""
Contains the custom functions needed to estimate the CDF/Quantile of the NTS distribution using custom integral method
"""

# Load modules
import numpy as np
import scipy.special as sc
from scipy.stats import norm
from sntn import type_farray

def _log_diff(log_p, log_q):
    return sc.logsumexp([log_p, log_q+np.pi*1j], axis=0)

def _log_gauss_mass(a, b):
    return _log_diff(sc.log_ndtr(b), sc.log_ndtr(a))


def Phi_diff(ub: type_farray, lb: type_farray) -> type_farray:
     """
     Calculates the difference between Gaussian CDFs:
     Phi(ub) - Phi(lb), where ub > lb
     """
     return np.abs(np.exp(_log_gauss_mass(ub, lb).real))
     

def bvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray):
    """
    Calculates the difference in the CDF between two bivariate normals with a shared x1 and rho value, by using the fact that: BVN(rho).cdf(x1, x2a) - BVN(rho).cdf(x1, x2b) = int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz, and that:

    (see 2.1.2 from https://arxiv.org/pdf/2006.03459)
    int Phi(a + bz) phi(z) dz = \
        0.5[Phi(z) + Phi(a/sqrt(1+b2)) - I(a / (z*sqrt(1+b^2)) > 2)]  - \
        [T(z, (a+bz)/z) - T(a/sqrt(1+b^2), ab + z(1+b^2))]
    where a = x1/sqrt(1+rho^2) & b = -rho/sqrt(1+rho^2), and therefore:
    a/sqrt(1+b^2)=x1 & (1+b^2)/a=[x1*sqrt(1-rho^2)]^{-1}

    Then if I(u; a, b) =  int_{-infty}^u Phi(a + bz) phi(z) dz, and 
    D(u, l; a, b) = I(u; a, b) - I(l; a, b) [i.e. the difference in binomial CDFs]
    D(x2a, x2b; a, b) = D(x2a, x2b; x1, rho) = \
    0.5 * { [Phi(x2b)-Phi(x2a)] - [I(x1/x2a<0) - I(x1/x2b<0)] } - \
    {[T(x2a, (x1/x2a-rho)/sqrt(1-rho^2)) - T(x2b, (x1/x2b-rho)/sqrt(1-rho^2)) ] + \
     [T(x1,(x2a/x1-rho)/sqrt(1-rho^2)) - T(x1,(x2b/x1-rho)/sqrt(1-rho^2))] }
    """
    rootrho = np.sqrt(1 - rho**2)
    den1a = (x1/x2a-rho) / rootrho
    den1b = (x1/x2b-rho) / rootrho
    den2a = (x2a/x1-rho) / rootrho
    den2b = (x2b/x1-rho) / rootrho
    term1_a = norm.cdf(x2a) - norm.cdf(x2b)
    term1_b = np.array(x1 / x2a < 0).astype(int) - np.array(x1 / x2b < 0).astype(int)
    term2_a = sc.owens_t(x2a, den1a) - sc.owens_t(x2b, den1b)
    term2_b = sc.owens_t(x1, den2a) - sc.owens_t(x1, den2b)
    pval = 0.5*(term1_a - term1_b) - (term2_a + term2_b)
    return pval


def _integrand_X12(x1: type_farray, x2: type_farray, rho: type_farray) -> type_farray:
    """See bvn_cdf_diff"""
    return norm.cdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)


# def _indicator(ineq: type_farray) -> bool:
#     if isinstance(ineq, np.ndarray):
#           return ineq.astype(int)
#     else:
#          return int(ineq)

# def _bvn_integral(z: type_farray, a: type_farray, b: type_farray) -> type_farray:
#      rootb = np.sqrt(1 + b**2)
#      beta = np.where(a / (z * rootb) < 0, 1, 0)
#      term1 = norm.cdf(z) + norm.cdf(a / rootb) - beta
#      term2 = owens_t(z, a/z + b) + owens_t(a/rootb, b + z*(1 + b**2)/a)
#      res = 0.5 * term1 - term2
#      return res


def bvn_cdf_diff_trapz(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray, n_points: int=1001) -> float | np.ndarray:
    """
    Calculates the difference in the CDF between two bivariate normals with a shared x1 and rho value:

    BVN(rho).cdf(x1, x2a) - BVN(rho).cdf(x1, x2b)

    Background
    ==========
    BVN(rho).cdf(x1, x2) = int_{-infty}^{x2} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    Since X1 | X2=z ~ N(rho*z, 1-rho^2)

    So the difference in the integrals is simply:
    int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    Parameters:
        - x1: np.ndarray or float, fixed x1 coordinate.
        - x2a, x2b: np.ndarray or float, the range [x2b, x2a] to integrate over.
        - rho: np.ndarray or float, correlation coefficient.
        - n_points: int, number of points for numerical integration.

    Returns:
        - float or np.ndarray: the difference in the CDF values.
    """
    x2a, x2b, x1, rho = np.broadcast_arrays(x2a, x2b, x1, rho)
    # Generate points and calculate dx
    points = np.linspace(x2b, x2a, num=n_points)  # points shape is now (n_points, *x2a.shape)
    dx = (x2a - x2b) / (n_points - 1)
    # Calculate integrand across the new axis
    integrand = _integrand_X12(x1, points, rho)
    # Integrate over the first axis, which is the points axis
    integral = np.trapz(integrand, dx=dx, axis=0)
    return integral  # integral = np.trapz(integrand, x=points, axis=-1)
    

def _integral110(z: type_farray, a: type_farray, b: type_farray) -> type_farray:
        """See """
        fb = np.sqrt(1 + b**2)
        res = (1/fb) * norm.pdf(a / fb) * norm.cdf(z*fb + a*b/fb)
        return res


def _dbvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray) -> type_farray:
    """
    See dbvn_cdf_diff before simplifcation
    """

    frho = np.sqrt(1-rho**2)
    a = x1 / frho
    b = -rho / frho
    val = (_integral110(x2a, a, b) - _integral110(x2b, a, b)) / frho
    return val


def dbvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray) -> type_farray:
    """
    Calculates the derivative of the integral:

    I(x1; x2a, x2b, rho) = int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    w.r.t x1:
    
    dI/dx1 = 1/sqrt(1-rho^2) int_{x2b}^{x2a} phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz
           = 1/sqrt(1-rho^2) [ (1/t)*phi(a/t)*Phi(tz + ab/t) ]|_2xb^x2a
           where a = x1/sqrt(1-rho^2), b = -rho/sqrt(1-rho^2), and t = sqrt(1 + b^2)

    From the closed form solution (see https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions and Owen 1980)

    But after simplifcation, this simply reduces to:
            = phi(x1) [Phi(z-x1*rho)]|_2xb^x2a
    """
    sigma_rho = np.sqrt(1-rho**2)
    ub = (x2a - x1*rho) / sigma_rho
    lb = (x2b - x1*rho) / sigma_rho
    val = norm.pdf(x1) * Phi_diff(ub, lb)
    return val


def d2bvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray) -> type_farray:
    """
    Calculates the derivative of the integral:

    D(x1; x2a, x2b, rho) = 1/sqrt(1-rho^2) int_{x2b}^{x2a} phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz
    
    w.r.t x1:
    
    D(x1; x2a, x2b, rho) = dD/dx1
        = -(1-rho^2)^{-1.5} int_{x2b}^{x2a} (x1-rho*z) phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz
        ... 

    since dphi(x; mu, sigma) = -phi(x; mu, sigma) * [(x-mu)/sigma^2]
    """
    return None