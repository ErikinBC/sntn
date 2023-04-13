"""
Checks that different approaches to bivariate normal...

python3 -m tests.test_grad
python3 -m pytest tests/test_grad.py -s
"""

# External modules
import pytest
import numpy as np
from scipy.stats import norm
from sntn.utilities.utils import vprint
from sntn.utilities.grad import _log_gauss_approx, _log_diff


# Parameters to test
x = [0.5, -0.0691, -0.0479, 0.4070]
a = [-1, -0.6235, -0.2974, -0.2679]
b = [1, 1.3765, 1.7026, 1.7321]
mu = [2, -0.0127, 0.0773, 0.3190]
sigma2 = [1.5, 1.1264, 0.1964, 1.0431]
testdata = np.vstack([x, a, b, mu, sigma2]).T
testdata = [list(testdata[j]) for j in range(4)]
fun_arg_ord = "x,a,b,mu,sigma2"
@pytest.mark.parametrize(fun_arg_ord, testdata)
def test_tnorm_grad(mu, x, a, b, sigma2, verbose:bool=False, tol:float=1e-10):
    """Make sure that the approximation for log(f(a)+f(b)) ~ _log_gauss_approx(f(a),f(b))"""
    sigma = [sigma2 ** 0.5]  # Will force array broadcast 
    x, a, b, mu, sigma = np.broadcast_arrays(x, a, b, mu, sigma)
    x_z = (x-mu)/sigma
    b_z = (b-mu)/sigma
    a_z = (a-mu)/sigma

    # --- Calculate the derivative with base functions --- #
    term1a = (norm.pdf(a_z)-norm.pdf(x_z))/sigma
    term1b = norm.cdf(b_z) - norm.cdf(a_z)
    term2a = norm.cdf(x_z) - norm.cdf(a_z)
    term2b = (norm.pdf(a_z)-norm.pdf(b_z))/sigma
    num = term1a*term1b - term2a*term2b
    denom = term1b**2
    val = num / denom  # quotient rule

    # --- Calculate the derivative with apprimxations --- #
    log_term1a = _log_gauss_approx(x_z, a_z, False) - np.log(sigma)
    log_term1b = _log_gauss_approx(b_z, a_z, True)
    log_term2a = _log_gauss_approx(x_z, a_z, True)
    log_term2b = _log_gauss_approx(b_z, a_z, False) - np.log(sigma)
    # The numerator will be off when the terms 1/2 do not align
    sign1a = np.where(np.abs(a_z) > np.abs(x_z),-1,+1)
    sign1b = np.where(b_z > a_z,+1,-1)
    term1_signs = sign1a * sign1b
    sign2a = np.where(x_z > a_z,+1,-1)
    sign2b = np.where(np.abs(a_z) > np.abs(b_z),-1,+1)
    term2_signs = sign2a * sign2b
    sign_fail = np.where(term1_signs != term2_signs, -1, +1) 
    log_fgprime = log_term1a + log_term1b
    log_gfprime = log_term2a + log_term2b
    log_num = np.real(_log_diff(log_fgprime, log_gfprime, sign_fail))
    log_denom = 2*_log_gauss_approx(b_z, a_z,True)
    val_approx = -np.exp(log_num - log_denom)

    # --- Check floating point differences --- #
    assert np.max(np.abs(term1a) - np.exp(log_term1a)) < tol
    assert np.max(np.abs(term1b) - np.exp(log_term1b)) < tol
    assert np.max(np.abs(term2a) - np.exp(log_term2a)) < tol
    assert np.max(np.abs(term2b) - np.exp(log_term2b)) < tol
    assert np.max(np.abs(num) - np.exp(log_num)) < tol
    assert np.max(np.abs(denom) - np.exp(log_denom)) < tol
    assert np.max(np.abs(val - val_approx)) < tol

    # Print if desired
    for j in range(len(mu)):
        vprint(f'---- {j} ----',verbose)
        vprint(f'term1a={term1a[j]:.6f}, approx={np.exp(log_term1a[j]):.6f}',verbose)
        vprint(f'term1b={term1b[j]:.6f}, approx={np.exp(log_term1b[j]):.6f}',verbose)
        vprint(f'term2a={term2a[j]:.6f}, approx={np.exp(log_term2a[j]):.6f}',verbose)
        vprint(f'term2b={term2b[j]:.6f}, approx={np.exp(log_term2b[j]):.6f}',verbose)
        vprint(f'numerator={num[j]:.6f}, approx={np.exp(log_num[j]):.6f}',verbose)
        vprint(f'denominator={denom[j]:.6f}, approx={np.exp(log_denom[j]):.6f}',verbose)
        vprint(f'Grad={val[j]}, approx={val_approx[j]}',verbose)
        vprint(f'Absolute error: {np.max(np.abs(val[j] - val_approx[j]))}',verbose)
        vprint('\n',verbose)



if __name__ == "__main__":
    for data in testdata:
        di_data = dict(zip(fun_arg_ord.split(','),data))
        test_tnorm_grad(**di_data)
    

    print('~~~ The test_bivariate.py script worked successfully ~~~')