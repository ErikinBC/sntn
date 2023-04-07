"""
Make sure distributions has expected properties to some reasonable tolerance

python3 -m tests.test_dists
python3 -m pytest tests/test_dists.py
"""

import pytest
import numpy as np
from scipy.stats import norm
from sntn.dists import tnorm
from parameters import seed

# Set seed
np.random.seed(seed)

params_tnorm = [((10, )), ((10, 5)), ((10, 5, 2)),]
@pytest.mark.parametrize("shape", params_tnorm)
def test_tnorm(shape, nsim=100000, tol=1e-2) -> None:
    # Check that the simulation obtains emprical moments we expect!
    mu = np.random.randn(*shape)
    sigma2 = np.exp(mu)
    a, b = mu - 2, mu + 2
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    Z = (norm.pdf(dist.alpha)-norm.pdf(dist.beta))/(norm.cdf(dist.beta)-norm.cdf(dist.alpha))
    mu_theory = dist.mu + Z*dist.sigma
    med_theory = dist.mu + norm.ppf((norm.cdf(dist.beta)+norm.cdf(dist.alpha))/2)*dist.sigma
    ndec = int(-np.log10(tol))
    err_mu = np.round(np.max(np.abs(np.mean(x, 0) - mu_theory)),ndec)
    err_med = np.round(np.max(np.abs(np.median(x, 0) - med_theory)),ndec)
    assert err_mu <= tol, f'Expected mean error to be less than {tol}: {err_mu} for {ndec} decimal places'
    assert err_mu <= tol, f'Expected median error to be less than {tol}: {err_med} for {ndec} decimal places' 


if __name__ == "__main__":
    # Check all functions
    for param in params_tnorm:
        test_tnorm(param)

    print('~~~ The test_dists.py script worked successfully ~~~')