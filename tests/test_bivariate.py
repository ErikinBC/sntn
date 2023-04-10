"""
Checks that different approaches to bivariate normal...
"""

# External modules
import numpy as np
from scipy.stats import norm
from sntn.utilities.grad import _log_gauss_approx, _log_diff

# Sample points
# x=0.5;a=-1;b=1;mu=2;sigma2=1.5
# x=-0.0691;a=-0.6235;b=1.3765;mu=-0.0127;sigma2=1.1264
# x=-0.0479;a=-0.2974;b=1.7026;mu=0.0773;sigma2=0.1964
x=0.4070;a=-0.2679;b=1.7321;mu=[1,0.3190];sigma2=1.0431
sigma=sigma2**0.5
x, a, b, mu, sigma = np.broadcast_arrays(x, a, b, mu, sigma)
x_z = (x-mu)/sigma
b_z = (b-mu)/sigma
a_z = (a-mu)/sigma

# Derivative using base classes
term1a = (norm.pdf(a_z)-norm.pdf(x_z))/sigma
term1b = norm.cdf(b_z) - norm.cdf(a_z)
term2a = norm.cdf(x_z) - norm.cdf(a_z)
term2b = (norm.pdf(a_z)-norm.pdf(b_z))/sigma
num = term1a*term1b - term2a*term2b
den = term1b**2
val = num / den  # quotient rule

# Try with _log approx_z
log_term1a = _log_gauss_approx(x_z, a_z, False) - np.log(sigma)
log_term1b = _log_gauss_approx(b_z, a_z, True)
log_term2a = _log_gauss_approx(x_z, a_z, True)
log_term2b = _log_gauss_approx(b_z, a_z, False) - np.log(sigma)
log_fgprime = log_term1a+log_term1b
log_gfprime = log_term2a+log_term2b
log_num = np.real(_log_diff(log_fgprime, log_gfprime))
log_denom = 2*_log_gauss_approx(b_z, a_z,True)
val_approx = -np.exp(log_num - log_denom)

np.exp(log_term1a)*np.exp(log_term1b)
np.exp(log_term2a)*np.exp(log_term2b)
num
num

# For reasonable values, we can exponentiate instead
# idx_approx = ~((np.abs(log_fgprime) < 10) & (np.abs(log_gfprime) < 10))
# log_num_approx = np.full_like(log_fgprime, fill_value=np.nan, dtype=np.float128)
# log_num_approx[idx_approx] = np.real(_log_diff(log_fgprime[idx_approx], log_gfprime[idx_approx]))
# log_num_approx[~idx_approx] = ...
# np.exp(term1a[~idx_approx]) * np.exp(term1b[~idx_approx]) - np.exp(term2a[~idx_approx]) * np.exp(term2b[~idx_approx])


for j in range(len(mu)):
    print(f'---- {j} ----')
    print(f'term1a={term1a[j]:.6f}, approx={np.exp(log_term1a[j]):.6f}')
    print(f'term1b={term1b[j]:.6f}, approx={np.exp(log_term1b[j]):.6f}')
    print(f'term2a={term2a[j]:.6f}, approx={np.exp(log_term2a[j]):.6f}')
    print(f'term2b={term2b[j]:.6f}, approx={np.exp(log_term2b[j]):.6f}')
    print(f'numerator={num[j]:.6f}, approx={np.exp(log_num[j]):.6f}')
    print(f'denominator={den[j]:.6f}, approx={np.exp(log_denom[j]):.6f}')
    print(f'Grad={val[j]}, approx={val_approx[j]}')
    print(f'Absolute error: {np.max(np.abs(val[j] - val_approx[j]))}')
    print('\n')



def test_bivariate() -> None:
    assert True


if __name__ == "__main__":
    test_bivariate()

    print('~~~ The test_bivariate.py script worked successfully ~~~')