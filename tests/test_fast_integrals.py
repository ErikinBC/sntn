"""
Make sure that the fast_integrals are getting the expected values

python3 -m pytest tests/test_fast_integrals.py -s
"""

import pytest
import numpy as np
from scipy.stats import norm, uniform
from numpy.testing import assert_allclose
from sntn._fast_integrals import Phi_diff

# Hard-code some parameters
atol = 1e-12
seed = 1234
nsim = 5692
params_atol = [atol,]
params_seed = [seed,]
params_nsim = [nsim,]

@pytest.mark.parametrize("atol", params_atol)
@pytest.mark.parametrize("seed", params_seed)
@pytest.mark.parametrize("nsim", params_nsim)
def test_Phi_diff(atol:float, nsim:int, seed:int) -> None:
    """
    Make sure that Phi(beta)-Phi(alpha) == Phi_diff(beta,alpha)
    """
    di_rvs = {'size':nsim, 'random_state':seed}
    beta = norm.rvs(**di_rvs)
    alpha = beta - uniform(loc=1, scale=1).rvs(**di_rvs)
    diff1 = norm.cdf(beta) - norm.cdf(alpha)
    diff2 = Phi_diff(beta, alpha)
    assert_allclose(actual=diff2, desired=diff1, atol=atol)
    print(f'\nMaximum error for Phi_diff = {np.abs(diff1 - diff2).max()} (size={nsim})\n')


if __name__ == "__main__":
    # Set the absolute tolerance limit

    test_Phi_diff(atol, nsim, seed)
    