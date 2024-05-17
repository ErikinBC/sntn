"""
Contains the custom functions needed to estimate the CDF/Quantile of the NTS distribution using custom integral method
"""

# Load modules
import numpy as np
import scipy.special as sc
from warnings import warn
from scipy.stats import norm
from scipy.optimize import newton
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
    term1_a = Phi_diff(x2a, x2b)
    term1_b = np.array(x1 / x2a < 0).astype(int) - np.array(x1 / x2b < 0).astype(int)
    # term1_b = (np.sign(x1) == -1).astype(int) - np.array(x1 / x2b < 0).astype(int)
    term2_a = sc.owens_t(x2a, den1a) - sc.owens_t(x2b, den1b)
    term2_b = sc.owens_t(x1, den2a) - sc.owens_t(x1, den2b)
    pval = 0.5*(term1_a - term1_b) - (term2_a + term2_b)
    return pval


def _dbvn_cdf_diff(x1: type_farray, x2a: type_farray, x2b: type_farray, rho: type_farray) -> type_farray:
    """
    See dbvn_cdf_diff before simplifcation
    """
    def _integral110(z: type_farray, a: type_farray, b: type_farray) -> type_farray:
        """See """
        fb = np.sqrt(1 + b**2)
        res = (1/fb) * norm.pdf(a / fb) * norm.cdf(z*fb + a*b/fb)
        return res

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
            = phi(x1) [Phi((z-x1*rho)/sqrt(1-rho^2))]|_2xb^x2a
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
                         = phi(x1) [Phi((z-x1*rho)/sqrt(1-rho^2))]|_2xb^x2a
    
    w.r.t x1:
    
    D2(x1; x2a, x2b, rho) = dD/dx1
        = -phi(x1) [x1 * Phi((z-x1*rho)/sqrt(1-rho^2)) + phi((z-x1*rho)/sqrt(1-rho^2))]|_2xb^x2a
    """
    sigma_rho = np.sqrt(1-rho**2)
    ub = (x2a - x1*rho) / sigma_rho
    lb = (x2b - x1*rho) / sigma_rho
    val = -norm.pdf(x1) * ( x1 * Phi_diff(ub, lb) + (rho/sigma_rho) * (norm.pdf(ub) - norm.pdf(lb)) )
    return val


def _modified_sigmoid(x, ub:float=5, slope:float=1, lb:float=0.1):
        """
        Function to bound the output to ±ub with the smallest absolute value of ±lb, takes on a sigmoid-like shape, with the high the slope, the more quickly we get to the ub
        """
        return np.sign(x) * (ub * np.tanh(slope * np.abs(x)) + lb)

def _grad_clip(grad:np.ndarray, clip_low: float=1e-4, clip_high: float=5):
    """Wrapper to do gradient clipping to keep away from zero or possibly a high number"""
    grad_clip = np.sign(grad) * np.clip(np.abs(grad), clip_low, clip_high)
    return grad_clip


def _rootfinder_newton(
          ub: type_farray, 
          lb: type_farray,
          rho: type_farray,
          target_p: type_farray,
          x0_vec: type_farray = -1.0, 
          use_hess: bool = True,
          use_gradclip: bool = False,
          clip_low: float = 1e-1, 
          clip_high: float = 5,
          tol: float=1e-8,
          ret_rootfun: bool=False
        ) -> np.ndarray:
    """
    Wrapper to support finding the roots of quantile for the SNTN distribution:
    {m: BVN(m,alpha,rho) - BVN(m, beta, rho) ~= alpha*Z }

    Inputs
    ======
    ub: type_farray
        The alpha > beta
    lb: type_farray
        The beta < alpha
    rho: type_farray
        The correlation coefficient
    target_p: type_farray
        The value of the binomial CDF difference to target
    x0_vec: Optional[float]
        The value to start the root finding algorithm at
    use_hess: bool = True
        Whether the second-order information should be used
    clip_low: float = 1e-2
        Smallest absolute gradient allowed
    clip_high: float = 5
        Largest absolute gradient allowed
    Returns
    =======
    The root-finding m-value which will need to be transformed to get to the original scale since m(z)=(z-theta)/sigma
    """
    # Set up the functions
    _rootfun = lambda z, ub, lb, rho, p: bvn_cdf_diff(z, ub, lb, rho) - p
    _drootfun = lambda z, ub, lb, rho, _: dbvn_cdf_diff(z, ub, lb, rho)
    if use_gradclip:
        _drootfun = lambda z, ub, lb, rho, _: _grad_clip(dbvn_cdf_diff(z, ub, lb, rho), clip_low, clip_high)
    _d2rootfun = lambda z, ub, lb, rho, _: d2bvn_cdf_diff(z, ub, lb, rho)
    # Make sure values are broadcasted
    x0_vec, ub, lb, rho, target_p = np.broadcast_arrays(x0_vec, ub, lb, rho, target_p)
    # Prepare function arguments
    di_newton = {'func':_rootfun, 'x0':x0_vec, 'fprime':_drootfun, 'args':(ub, lb, rho, target_p)}
    if use_hess:
       di_newton['fprime2'] = _d2rootfun
    roots = newton(**di_newton)
    y = _rootfun(roots, ub, lb, rho, target_p)
    mx_err = np.abs(y).max()
    if mx_err > tol:
         warn(f'One or more roots did non convergence to within {tol} = {mx_err}')
    if ret_rootfun:
        return roots, y
    else:
         return roots
