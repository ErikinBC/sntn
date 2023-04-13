"""
Utility functions for gradients
"""

# Load modules
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr, ndtr, log1p


####################################
# ---- (1) TRUNCATED GAUSSIAN ---- #

def _log_diff(log_p, log_q, signs:None or np.ndarray=None):
    return _logsumexp(a=log_p, b=log_q+np.pi*1j, signs=signs)

# CDF
def _case_left_cdf(a, b):
    val1, val2 = log_ndtr(b), log_ndtr(a)
    return _log_diff(val1, val2)

def _case_right_cdf(a, b):
    return _case_left_cdf(-b, -a)

def _case_central_cdf(a, b):
    val = -ndtr(a) - ndtr(-b)
    return log1p(val)

# PDF
def _case_left_pdf(a, b):
    val1, val2 = norm.logpdf(b), norm.logpdf(a)
    return _log_diff(val1, val2)

def _case_right_pdf(a, b):
    return _case_left_pdf(-b, -a)

def _case_central_pdf(a, b):
    val = -norm.pdf(a) - norm.pdf(-b)
    return log1p(val)


def _logsumexp(a:np.ndarray, b:np.ndarray, signs:None or np.ndarray=None):
    """Calculates the log of sum of exponentials: log(a[i] + signs[i]*b[i])

    Parameters
    ----------
    a : array_like
        Input array.
    b : array_like
        Input array.
    signs : array_like
         Modifies the sum (which can through an error if it is negative)

    Returns
    -------
    res : ndarray
    """
    if signs is not None:
        a, b, signs = np.broadcast_arrays(a, b, signs)
    else:
        a, b = np.broadcast_arrays(a, b)
        signs = np.ones(a.shape)
    # Check sizes
    assert a.shape == b.shape == signs.shape, 'a/b/signs need to have the same shape'
    # Flatten for later
    is_flat = False
    if a.ndim > 1:
        is_flat = True
        store_shape = a.shape
        a, b, signs = a.flatten(), b.flatten(), signs.flatten()
    mat = np.c_[a, b]
    mat_max = np.max(mat, axis=1, keepdims=True)
    mat_max[~np.isfinite(mat_max)] = 0
    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        mat_adj = np.exp(mat - mat_max)
        s = mat_adj[:,0] + signs*mat_adj[:,1]
        out = np.log(s)
    out += mat_max.flat  # Add back on constant
    if is_flat:
        out = out.reshape(store_shape)
    return out


def _log_gauss_approx(a:np.ndarray, b:np.ndarray, cdf:bool=True) -> np.ndarray:
    """Log of Gaussian probability mass within an interval"""
    # Process inputs
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    a, b = np.broadcast_arrays(a, b)
    case_left = b <= 0
    case_right = a > 0
    case_central = ~(case_left | case_right)
    # Left
    fun_left = _case_left_cdf
    if not cdf:
        fun_left = _case_left_pdf
    # Right
    fun_right = _case_right_cdf
    if not cdf:
        fun_right = _case_right_pdf
    # Central
    fun_central = _case_central_cdf
    if not cdf:
        fun_central = _case_central_pdf
    # Get value
    out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
    if a[case_left].size:
        out[case_left] = fun_left(a[case_left], b[case_left])
    if a[case_right].size:
        out[case_right] = fun_right(a[case_right], b[case_right])
    if a[case_central].size:
        out[case_central] = fun_central(a[case_central], b[case_central])
    return np.real(out)
