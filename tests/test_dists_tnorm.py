"""
Make sure truncated normal wrapper (tnorm) has expected properties to some reasonable tolerance

python3 -m tests.test_dists_tnorm
python3 -m pytest tests/test_dists_tnorm.py
"""

import pytest
import numpy as np
from scipy.stats import norm, kstest
# Internal
from parameters import seed
from sntn.dists import tnorm

def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates TN parameters"""
    np.random.seed(seed)
    mu = np.random.randn(*shape)
    sigma2 = np.exp(np.random.randn(*shape))
    a = np.random.randn(*shape)
    b = a + 2
    return mu, sigma2, a, b


params_tnorm_rvs = [((10, )), ((10, 5)), ((10, 5, 2)),]
@pytest.mark.parametrize("shape", params_tnorm_rvs)
def test_tnorm_rvs(shape:tuple or list, nsim:int=100000, tol1:float=1e-2, tol2:float=1e-9) -> None:
    """Check that rvs() gets the mean/median we expect from theory"""
    # Process inputs
    mu, sigma2, a, b = gen_params(shape, seed)
    dist = tnorm(mu, sigma2, a, b)
    ndec = int(-np.log10(tol1))
    # Check that theory lines up with underlying scipy dist
    x = dist.rvs(nsim, seed)
    Z = (norm.pdf(dist.alpha)-norm.pdf(dist.beta))/(norm.cdf(dist.beta)-norm.cdf(dist.alpha))
    mu_theory = dist.mu + Z*dist.sigma
    med_theory = dist.mu + norm.ppf((norm.cdf(dist.beta)+norm.cdf(dist.alpha))/2)*dist.sigma
    assert np.all(np.abs(dist.dist.mean() - mu_theory) < tol2)
    assert np.all(np.abs(dist.dist.median() - med_theory) < tol2)
    # Compare to the simulated data
    err_mu = np.round(np.max(np.abs(np.mean(x, 0) - mu_theory)),ndec)
    err_med = np.round(np.max(np.abs(np.median(x, 0) - dist.dist.median())),ndec)
    assert err_mu <= tol1, f'Expected mean error to be less than {tol1}: {err_mu} for {ndec} decimal places'
    assert err_mu <= tol1, f'Expected median error to be less than {tol1}: {err_med} for {ndec} decimal places' 


def test_tnorm_cdf(n:int=1, nsim:int=10000, alpha:float=0.05) -> None:
    """Check that the cdf is uniform"""
    mu, sigma2, a, b = gen_params((n,), seed)
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    pval = kstest(x, dist.cdf, method='exact').pvalue
    assert pval > alpha, f'kstest rejected null of uniformity'
    pval_err = kstest(x+1, dist.cdf, method='exact').pvalue
    assert pval_err < alpha, f'kstest did not reject null of uniformity'


def test_tnorm_ppf(n:int=1, nsim:int=1000000, tol:float=1e-2) -> None:
    """Check that the q/q plot is linear with a slope of 1"""
    mu, sigma2, a, b = gen_params((n,), seed)
    dist = tnorm(mu, sigma2, a, b)
    x = dist.rvs(nsim, seed)
    pseq = np.linspace(0.01, 0.99, 99)
    q_emp = np.quantile(x, pseq)
    q_theory = dist.ppf(pseq)
    err_q = np.max(np.abs(q_emp - q_theory))
    assert err_q < tol, f'QQ error was greater than {tol}: {err_q}'


params_tnorm_fit = [((1, )), ((5,)), ((3, 2)),]
@pytest.mark.parametrize('n', params_tnorm_fit)
# @pytest.mark.parametrize('use_sigma', [(True)])
def test_tnorm_fit(n:int, use_sigma:bool=True, nsim:int=50000, tol:float=1e-3) -> None:
    """Checks that with sufficiently large samples we get point estimates"""
    # Generate data
    mu, sigma2, a, b = gen_params(n, seed)
    a = np.where(a > mu, a - np.abs(a - mu) - 1, a)
    b = np.where(b < mu, b + np.abs(b - mu) + 1, b)
    oracle_dist = tnorm(mu, sigma2, a, b)
    samp = oracle_dist.rvs(nsim, seed=seed)
    # Get mu_hat fit
    _, _, mu_hat, _ = oracle_dist.fit(samp, use_a=True, use_b=True, use_sigma=use_sigma)
    err = np.abs(mu_hat - mu)
    idx_fail = err > tol
    mx_err = np.max(np.abs(mu_hat - mu))
    assert mx_err <= tol, f'Expected maximum error to be less than {tol}: {mx_err} ({mu[idx_fail][0], sigma2[idx_fail][0], a[idx_fail][0], b[idx_fail][0]})'


def test_tnorm_CIs(n:int=1, ndraw:int=10) -> None:
    """Check that the confidence interval is working as expected"""
    # Generate data
    mu, sigma2, a, b = gen_params((n,), seed)
    dist = tnorm(mu, sigma2, a, b)
    # Generate data
    x = dist.rvs(ndraw, seed)

    # # (i) "root_scalar" apprach
    # methods_root_scalar = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748', 'newton', 'secant', 'halley']
    # for method in methods_root_scalar:
    #     print(f'Testing method {method} for root_scalar')
    #     res = dist.get_CI(x=x, approach='root_scalar', method=method)

    # # (ii) "minimizer_scalar" appraoch
    # methods_minimize_scalar = ['Brent', 'Bounded', 'Golden']
    # for method in methods_minimize_scalar:
    #     print(f'Testing method {method} for minimize_scalar')
        # res = dist.get_CI(x=x, approach='minimize_scalar', method=method)

    # (iii) "minimize" approach
    methods_minimize = ['Nelder-Mead', 'Powell', 'COBYLA']
    for method in methods_minimize:
        print(f'Testing method {method} for minimize')
        res = dist.get_CI(x=x, approach='minimize', method=method)
    
    # # (iv) Check "root" approach
    # methods_root = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
    # for method in methods_root:
    #     print(f'Testing method {method} for root')
    #     res = dist.get_CI(x=x, approach='root', method=method)



    


if __name__ == "__main__":
    # # Loop over rvs params
    # for param in params_tnorm_rvs:
    #     test_tnorm_rvs(param)
    # test_tnorm_cdf()
    # test_tnorm_ppf()
    test_tnorm_CIs()

    print('~~~ The test_dists.py script worked successfully ~~~')