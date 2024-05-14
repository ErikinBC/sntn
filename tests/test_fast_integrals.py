"""
Make sure that the fast_integrals are getting the expected values

python3 -m tests.test_fast_integrals
python3 -m pytest tests/test_fast_integrals.py -s
python3 -m pytest tests/test_fast_integrals.py -s -k 'test_dcdf_diff'
"""

# External
import pytest
import numpy as np
from typing import Tuple, Dict
from scipy.stats import norm, uniform
from scipy.optimize import check_grad
from numpy.testing import assert_allclose
# Internal
from sntn._bvn import _bvn as bvn
from sntn._fast_integrals import Phi_diff, bvn_cdf_diff, dbvn_cdf_diff

# Parameters configuration as a fixture
@pytest.fixture(scope="module", params=[(1e-7, 1234, 5692)])
def test_params(request):
    atol, seed, nsim = request.param
    return {'atol': atol, 'seed': seed, 'nsim': nsim}


def _return_arg_dicts(test_params) -> Tuple[float, dict, dict]:
    """Cleans up the test_params to be used for later"""
    atol, seed, nsim = test_params['atol'], test_params['seed'], test_params['nsim']
    di_size_vec = {'size': nsim, 'random_state': seed}
    di_size_mat = {'size': [nsim, 2], 'random_state': seed}
    return atol, di_size_vec, di_size_mat

def _gen_x1_x2ab_rho(
        size_matrix: Dict[str, int], 
        size_vector: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = norm.rvs(**size_matrix)
    x1, x2a = x.T
    x2b = x2a - uniform(loc=1, scale=1).rvs(**size_vector)
    rho = uniform.rvs(**size_vector)
    return x1, x2a, x2b, rho


def test_Phi_diff(test_params):
    """
    Make sure that Phi(beta)-Phi(alpha) == Phi_diff(beta,alpha)
    """
    atol, di_size_vec, _ = _return_arg_dicts(test_params)
    beta = norm.rvs(**di_size_vec)
    alpha = beta - uniform(loc=1, scale=1).rvs(**di_size_vec)
    diff1 = norm.cdf(beta) - norm.cdf(alpha)
    diff2 = Phi_diff(beta, alpha)
    assert_allclose(actual=diff2, desired=diff1, atol=atol)
    print(f'\nMaximum error for Phi_diff = {np.abs(diff1 - diff2).max()} (size={beta.shape[0]})\n')


def test_cdf_diff(test_params) -> None:
    """
    Make sure that 
    bvn_cdf_diff(x1, x2a, x2b, rho) == BVN(x1, x2a; rho) - BVN(x1, x2b; rho)
    """
    atol, di_size_vec, di_size_mat = _return_arg_dicts(test_params)
    x1, x2a, x2b, rho = _gen_x1_x2ab_rho(di_size_mat, di_size_vec)
    bvn_owen = bvn(0, 1, 0, 1, rho, cdf_approach='owen')
    pval_owen = bvn_owen.cdf(x1=x1, x2=x2a) - bvn_owen.cdf(x1=x1, x2=x2b)
    pval_diff = bvn_cdf_diff(x1=x1, x2a=x2a, x2b=x2b, rho=rho)
    assert_allclose(actual=pval_diff, desired=pval_owen, atol=atol)
    print(f'\nMaximum error for [Phi(a)-Phi(b)]-dPhi(a,b) = {np.abs(pval_diff - pval_owen).max()} (size={rho.shape[0]})\n')
    

def test_dcdf_diff(test_params) -> None:
    """
    Make sure we can accurately calculate the gradient of bvn_cdf_diff w.r.t x1
    """
    atol, di_size_vec, di_size_mat = _return_arg_dicts(test_params)
    x1, x2a, x2b, rho = _gen_x1_x2ab_rho(di_size_mat, di_size_vec)
    grad_diff = [check_grad(bvn_cdf_diff, dbvn_cdf_diff, x1_i, x2a_i, x2b_i, rho_i) for (x1_i, x2a_i, x2b_i, rho_i) in zip(x1, x2a, x2b, rho)]
    # grad_diff = [check_grad(func=lambda x: x**2, grad=lambda x: 2*x, x0=x0) for x0 in np.array([3,4])]
    assert_allclose(actual=grad_diff, desired=0, atol=atol)
    print(f'\nMaximum grad error for dPhi(a,b; x1, rho) = {np.abs(grad_diff).max()} (size={rho.shape[0]})\n')