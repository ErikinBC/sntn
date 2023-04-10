"""
Utility functions for gradients
"""

# Load modules
import numpy as np
import scipy.special as sc
from scipy.stats import norm

####################################
# ---- (1) TRUNCATED GAUSSIAN ---- #

def _log_diff(log_p, log_q):
    return sc.logsumexp([log_p, log_q+np.pi*1j], axis=0)

# CDF
def _case_left_cdf(a, b):
    return _log_diff(sc.log_ndtr(a), sc.log_ndtr(b))

def _case_right_cdf(a, b):
    return _case_left_cdf(-a, -b)

def _case_central_cdf(a, b):
    return sc.log1p(-sc.ndtr(b) - sc.ndtr(-a))

# PDF
def _case_left_pdf(a, b):
    return _log_diff(norm.logpdf(a), norm.logpdf(b))

def _case_right_pdf(a, b):
    return _case_left_pdf(-a, -b)

def _case_central_pdf(a, b):
    return sc.log1p(-norm.pdf(b) - norm.pdf(-a))


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
