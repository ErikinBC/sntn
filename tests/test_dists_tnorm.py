"""
Make sure truncated normal wrapper (tnorm) has expected properties to some reasonable tolerance

python3 -m tests.test_dists_tnorm
python3 -m pytest tests/test_dists_tnorm.py -s
"""

# External
import os
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm, kstest
# Internal
from sntn.dists import tnorm
from parameters import seed, dir_simulations
from sntn.utilities.utils import vprint, is_equal
from sntn._solvers import di_default_methods

# Parameters recycled
params_tnorm_rvs = [((1,)), ((10, )), ((10, 5)), ((10, 5, 2)),]


def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates TN parameters
    
    Returns
    -------
    (mu, sigma2, a, b)
    """
    np.random.seed(seed)
    mu = np.random.randn(*shape)
    sigma2 = np.exp(np.random.randn(*shape))
    a = np.random.randn(*shape)
    b = a + 2
    return mu, sigma2, a, b


@pytest.mark.parametrize("shape", params_tnorm_rvs)
def test_dmu(shape:tuple or list, eps:float=1e-6, tol:float=1e-7, verbose:bool=False):
    """Check that the numerical derivatives align with analytical ones"""
    mu, sigma2, a, b = gen_params(shape, seed)
    alpha, a_min, a_max = 0.05, None, None
    dist = tnorm(mu, sigma2, a, b)
    dist_plus = tnorm(mu+eps, sigma2, a, b)
    dist_minus = tnorm(mu-eps, sigma2, a, b)
    x = np.squeeze(dist.rvs(1, seed))  # Draw one sample from each
    # Calculate the numerical derivative
    dmu_num = (dist_plus.cdf(x)-dist_minus.cdf(x))/(2*eps)
    # Run the exact analytical derivative
    dmu_ana = dist._dmu_dcdf(mu, x, alpha, sigma2=sigma2, a=a, b=b, approx=False)
    # Run the log-approx derivative
    dmu_approx = dist._dmu_dcdf(mu=mu, x=x, sigma2=sigma2, a=a, b=b, alpha=alpha, approx=True)
    # Check for differences
    vprint(f'Largest error b/w analytic and exact : {np.max(np.abs(dmu_ana - dmu_num)):.12f}',verbose)
    vprint(f'Largest error b/w exact and approx : {np.max(np.abs(dmu_approx - dmu_num)):.12f}',verbose)
    vprint(f'Largest error b/w analytic and approx : {np.max(np.abs(dmu_ana - dmu_approx)):.12f}',verbose)
    is_equal(dmu_ana, dmu_num, tol)
    is_equal(dmu_num, dmu_approx, tol)
    is_equal(dmu_ana, dmu_approx, tol)


@pytest.mark.parametrize("shape", params_tnorm_rvs)
def test_tnorm_rvs(shape:tuple or list, nsim:int=100000, tol1:float=1e-2, tol2:float=1e-9) -> None:
    """Check that rvs() gets the mean/median we expect from theory"""
    # Process inputs
    mu, sigma2, a, b = gen_params(shape, seed)
    dist = tnorm(mu, sigma2, a, b)
    ndec = int(-np.log10(tol1))
    # Check that theory lines up with underlying scipy dist
    x = dist.rvs(nsim, seed)
    Z = (norm.pdf(dist._truncnorm.alpha)-norm.pdf(dist._truncnorm.beta))/(norm.cdf(dist._truncnorm.beta)-norm.cdf(dist._truncnorm.alpha))
    mu_theory = dist._truncnorm.mu + Z*dist._truncnorm.sigma
    med_theory = dist._truncnorm.mu + norm.ppf((norm.cdf(dist._truncnorm.beta)+norm.cdf(dist._truncnorm.alpha))/2)*dist._truncnorm.sigma
    assert np.all(np.abs(dist._truncnorm.dist.mean() - mu_theory) < tol2)
    assert np.all(np.abs(dist._truncnorm.dist.median() - med_theory) < tol2)
    # Compare to the simulated data
    err_mu = np.round(np.max(np.abs(np.mean(x, 0) - mu_theory)),ndec)
    err_med = np.round(np.max(np.abs(np.median(x, 0) - dist._truncnorm.dist.median())),ndec)
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


params_CI = [ ((1,), 5), ((5,), 1), ((3,2), 1), ((2,2,2), 1) ]
@pytest.mark.parametrize('n,ndraw', params_CI)
def test_tnorm_CI(n, ndraw, alpha:float=0.05, approx:bool=True) -> None:
    """Check that the confidence interval is working as expected"""
    print(f'n={n}, ndraw={ndraw}')
    # Generate data
    mu, sigma2, a, b = gen_params(n, seed)
    dist = tnorm(mu, sigma2, a, b)
    # Generate data
    x = dist.rvs(ndraw, seed)

    # (i) "root_scalar" apprach (ignoring 'halley' since it requires Hessian)
    methods_root_scalar = di_default_methods['root_scalar']
    holder_root_scalar = []
    for method in methods_root_scalar:
        print(f'Testing method {method} for root_scalar')
        di_scipy = {'method':method}
        if method == 'newton':
            res = dist.conf_int(x=x, alpha=alpha, approach='root_scalar', di_scipy=di_scipy, approx=approx, sigma2=sigma2, a=a, b=b, a_min=1e-5, a_max=np.inf)    
        else:
            res = dist.conf_int(x=x, alpha=alpha, approach='root_scalar', di_scipy=di_scipy, approx=approx, sigma2=sigma2, a=a, b=b)
        if res.ndim > 2:
            res = res.reshape([int(np.prod(n)), 2])
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_root_scalar.append(res)
    res_root_scalar = pd.concat(holder_root_scalar).assign(approach='root_scalar')

    # (ii) "minimize_scalar" approach
    methods_minimize_scalar = di_default_methods['minimize_scalar']
    holder_minimize_scalar = []
    for method in methods_minimize_scalar:
        print(f'Testing method {method} for minimize_scalar')
        di_scipy = {'method':method}
        res = dist.conf_int(x=x, alpha=alpha, approach='minimize_scalar', di_scipy=di_scipy, approx=approx, sigma2=sigma2, a=a, b=b)
        if res.ndim > 2:
            res = res.reshape([int(np.prod(n)), 2])
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_minimize_scalar.append(res)
    res_minimize_scalar = pd.concat(holder_minimize_scalar).assign(approach='minimize_scalar')


    # (iii) "minimize" approach 
    methods_minimize = di_default_methods['minimize']
    holder_minimize = []
    for method in methods_minimize:
        print(f'Testing method {method} for minimize')
        di_scipy = {'method':method}
        res = dist.conf_int(x=x, alpha=alpha, approach='minimize', di_scipy=di_scipy, approx=approx, sigma2=sigma2, a=a, b=b)        
        if res.ndim > 2:
            res = res.reshape([int(np.prod(n)), 2])
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_minimize.append(res)
    res_minimize = pd.concat(holder_minimize).assign(approach='minimize')

    # (iv) Check "root" approach
    methods_root = di_default_methods['root']
    holder_root = []
    for method in methods_root:
        print(f'Testing method {method} for root')
        di_scipy = {'method':method}
        res = dist.conf_int(x=x, alpha=alpha, approach='root', di_scipy=di_scipy, approx=approx, sigma2=sigma2, a=a, b=b)
        if res.ndim > 2:
            res = res.reshape([int(np.prod(n)), 2])
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_root.append(res)
    res_root = pd.concat(holder_root).assign(approach='root')

    # Combine and save results
    res_all = pd.concat(objs=[res_root_scalar, res_minimize, res_minimize_scalar, res_root])
    # Determine the index
    x_flat, mu_flat, sigma2_flat, a_flat, b_flat = [z.flatten() for z in np.broadcast_arrays(x, mu, sigma2, a, b)]
    dat_mu_idx = pd.DataFrame({'idx':range(x_flat.shape[0]), 'x':x_flat, 'mu':mu_flat, 'sigma2':sigma2_flat, 'a':a_flat, 'b':b_flat})
    res_all = res_all.rename_axis('idx').reset_index().merge(dat_mu_idx)
    res_all = res_all.melt(np.setdiff1d(res_all.columns, ['lb','ub']),['lb','ub'],var_name='bound')
    res_all = res_all.assign(n=str(n), ndraw=ndraw)    
    # Clean up file name for saving
    fn_save = f"res_test_norm_CI_{'_'.join([str(i) for i in n])}_{ndraw}.csv"
    res_all.to_csv(os.path.join(dir_simulations, fn_save),index=False)


if __name__ == "__main__":
    print('--- test_tnorm_cdf ---')
    test_tnorm_cdf()

    print('--- test_tnorm_ppf ---')
    test_tnorm_ppf()

    print('--- test_tnorm_fit ---')
    for param in params_tnorm_fit:
        test_tnorm_fit(param, use_sigma=True)

    print('--- test_dmu ---')
    for param in params_tnorm_rvs:
        print(f'param={param}')
        test_dmu(param)

    print('--- test_tnorm_rvs ---')
    for param in params_tnorm_rvs:
        print(f'param={param}')
        test_tnorm_rvs(param)
    
    print('--- test_tnorm_CI ---')
    for param in params_CI:
        n, ndraw = param[0], param[1]
        test_tnorm_CI(n, ndraw) 

    print('~~~ The test_dists_tnorm.py script worked successfully ~~~')