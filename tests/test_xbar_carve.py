"""
Makes sure that the data carved power aligns with expectations for the simple case of the sample mean

python3 -m pytest tests/test_xbar_carve.py -s
"""

# External
import pytest
import numpy as np
from scipy.stats import norm
# Internal
from sntn.dists import nts, tnorm
from parameters import seed

frac_seq = [0.25, 0.50, 0.75]
@pytest.mark.parametrize("frac_split", frac_seq)
def test_xbar_carve(frac_split:float, nsim:int=1000000, alpha:float=0.1, a:float=1.0, b:float=np.inf, mu_null:float=0.0, mu_alt:float=0.5, sigma2:float=4, n:int=100, tol:float=9e-4) -> None:
    # Set up parameters
    n_split = int(n * frac_split)
    n_screen = n - n_split
    sd_split = np.sqrt(sigma2 / n_split)
    sd_screen = np.sqrt(sigma2 / n_screen)
    c1 = n_split / n
    c2 = n_screen / n

    # Use NTS class to generate critival values and cdf (power)
    nts_null_u = nts(mu_null, sd_split**2, mu_null, sd_screen**2, a, b, 1, 1)
    nts_null_w = nts(mu_null, sd_split**2, 0, sd_screen**2, a, b, c1, c2)
    nts_alt_u = nts(mu_alt, sd_split**2, mu_alt, sd_screen**2, a, b, 1, 1)
    nts_alt_w = nts(mu_alt, sd_split**2, mu_alt, sd_screen**2, a, b, c1, c2)

    # Critical values
    crit_v_u = nts_null_u.ppf(1-alpha)[0]
    crit_v_w = nts_null_w.ppf(1-alpha)[0]
    power_u = 1 - nts_alt_u.cdf(crit_v_u)[0]
    power_w = 1 - nts_alt_w.cdf(crit_v_w)[0]
    assert np.abs(power_u - np.mean(nts_alt_u.rvs(nsim, seed=seed) > crit_v_u)) < tol, 'cdf != rvs'
    assert np.abs(power_w - np.mean(nts_alt_w.rvs(nsim, seed=seed) > crit_v_w)) < tol, 'cdf != rvs'

    # Repeat with rvs
    dist_gauss_null = norm(mu_null, sd_split)
    dist_gauss_alt = norm(mu_alt, sd_split)
    rvs_null_u = dist_gauss_null.rvs(nsim,random_state=seed) + tnorm(mu_null,sd_screen**2, a, b).rvs(nsim,seed=seed)
    rvs_null_w = c1*dist_gauss_null.rvs(nsim,random_state=seed) + c2*tnorm(mu_null,sd_screen**2, a, b).rvs(nsim,seed=seed)
    rvs_alt_u = dist_gauss_alt.rvs(nsim,random_state=seed) + tnorm(mu_alt,sd_screen**2, a, b).rvs(nsim,seed=seed)
    rvs_alt_w = c1*dist_gauss_alt.rvs(nsim,random_state=seed) + c2*tnorm(mu_alt,sd_screen**2, a, b).rvs(nsim,seed=seed)
    emp_critv_u = np.quantile(rvs_null_u, 1-alpha)
    emp_power_u = np.mean(rvs_alt_u > emp_critv_u)
    emp_critv_w = np.quantile(rvs_null_w, 1-alpha)
    emp_power_w = np.mean(rvs_alt_w > emp_critv_w)

    # Calculate power for split-only
    crit_v_split = dist_gauss_null.ppf(1-alpha)
    power_split = 1 - dist_gauss_alt.cdf(crit_v_split)

    assert np.abs(power_u - emp_power_u) < tol
    assert np.abs(power_w - emp_power_w) < tol
    assert power_w >= power_u, f'Expected {power_w} to be greater than {power_u}'
    assert power_w > power_split, f'Expected {power_w} to be greater than {power_split}'

    # Compare results
    print(f'~~~ c1={c1:.2f} (n1={n_split}), c2={c2:.2f} (n2={n_screen}) ~~~')
    print(f'Unweighted power for theory={power_u*100:.2f}%, empirical={emp_power_u*100:.2f}%')
    print(f'Weighted power for theory={power_w*100:.2f}%, empirical={emp_power_w*100:.2f}%')
    print(f'Power from splitting only={power_split*100:.2f}%')
    