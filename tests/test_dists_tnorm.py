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

params_tnorm_rvs = [((1,)), ((10, )), ((10, 5)), ((10, 5, 2)),]

def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates TN parameters"""
    np.random.seed(seed)
    mu = np.random.randn(*shape)
    sigma2 = np.exp(np.random.randn(*shape))
    a = np.random.randn(*shape)
    b = a + 2
    return mu, sigma2, a, b

@pytest.mark.parametrize("shape", params_tnorm_rvs)
def test_dmu(shape:tuple or list, eps:float=1e-6, tol:float=1e-7):
    """Check that the numerical derivatives align with analytical ones"""
    mu, sigma2, a, b = gen_params(shape, seed)
    dist = tnorm(mu, sigma2, a, b)
    dist_plus = tnorm(mu+eps, sigma2, a, b)
    dist_minus = tnorm(mu-eps, sigma2, a, b)
    x = np.squeeze(dist.rvs(1, seed))  # Draw one sample from each
    # Calculate the numerical derivative
    dmu_num = (dist_plus.cdf(x)-dist_minus.cdf(x))/(2*eps)
    # Run the exact analytical derivative
    dmu_ana = dist._dmu_dcdf(mu, x, approx=False)
    # Run the log-approx derivative
    dmu_approx = dist._dmu_dcdf(mu, x, approx=True)
    # Check for differences
    # print(f'Largest error b/w analytic and exact : {np.max(np.abs(dmu_ana - dmu_num))}')
    # print(f'Largest error b/w exact and approx : {np.max(np.abs(dmu_approx - dmu_num))}')
    # print(f'Largest error b/w analytic and approx : {np.max(np.abs(dmu_ana - dmu_approx))}')
    if np.max(np.abs(dmu_approx - dmu_num)) > 0.01:
        idx_fail = np.argmax(np.abs(dmu_num-dmu_approx))
        try:
            print(f'x={x.flat[idx_fail]:.4f};a={a.flat[idx_fail]:.4f};b={b.flat[idx_fail]:.4f};mu={mu.flat[idx_fail]:.4f};sigma2={sigma2.flat[idx_fail]:.4f}')
        except:
            breakpoint()
    # is_equal(dmu_ana, dmu_num, tol)
    # is_equal(dmu_num, dmu_approx, tol)
    # is_equal(dmu_ana, dmu_approx, tol)
    
    



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


def test_tnorm_CI(n:int=1, ndraw:int=10) -> None:
    """Check that the confidence interval is working as expected"""
    # Generate data
    mu, sigma2, a, b = gen_params((n,), seed)
    dist = tnorm(mu, sigma2, a, b)
    # Generate data
    x = dist.rvs(ndraw, seed)[:2]

    # (i) "root_scalar" apprach ('newton', 'halley')
    methods_root_scalar = ['newton']  #'newton', 'bisect', 'brentq', 'brenth', 'ridder','toms748', 'secant'
    holder_root_scalar = []
    for method in methods_root_scalar:
        print(f'Testing method {method} for root_scalar')
        res = dist.get_CI(x=x, approach='root_scalar', method=method)
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_root_scalar.append(res)
    res_root_scalar = pd.concat(holder_root_scalar).assign(approach='root_scalar')

    # (ii) "minimizer_scalar" appraoch
    methods_minimize_scalar = ['Brent', 'Bounded', 'Golden']
    holder_minimize_scalar = []
    for method in methods_minimize_scalar:
        print(f'Testing method {method} for minimize_scalar')
        res = dist.get_CI(x=x, approach='minimize_scalar', method=method)
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_minimize_scalar.append(res)
    res_minimize_scalar = pd.concat(holder_minimize_scalar).assign(approach='minimize_scalar')

    # (iii) "minimize" approach
    methods_minimize = ['Nelder-Mead', 'Powell', 'COBYLA']
    holder_minimize = []
    for method in methods_minimize:
        print(f'Testing method {method} for minimize')
        res = dist.get_CI(x=x, approach='minimize', method=method)
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_minimize.append(res)
    res_minimize = pd.concat(holder_minimize).assign(approach='minimize')

    # (iv) Check "root" approach
    methods_root = ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']
    holder_root = []
    for method in methods_root:
        print(f'Testing method {method} for root')
        res = dist.get_CI(x=x, approach='root', method=method)
        res = pd.DataFrame(res,columns=['lb','ub']).assign(method=method)
        holder_root.append(res)
    res_root = pd.concat(holder_minimize).assign(approach='root')
    
    # Combine and save results
    res_all = pd.concat(objs=[res_root_scalar, res_minimize, res_minimize_scalar, res_root])
    res_all = res_all.rename_axis('idx').melt(['method','approach'],['lb','ub'],ignore_index=False,var_name='bound').reset_index()
    res_all['idx'] = pd.Categorical(res_all['idx'] + 1, range(1,ndraw+1))
    res_all = res_all.assign(num=lambda x: (pd.Categorical(x['method']).codes+1))
    res_all = res_all.assign(color = lambda x: x['num'].astype(str) + '.' + x['method'])
    # Add on the true parameters
    assert len(mu) == 1, 'Cannot assign if value > 1'
    res_all = res_all.assign(mu0=mu[0])
    res_all.to_csv(os.path.join(dir_simulations, 'res_test_norm_CI.csv'),index=False)


 
    

if __name__ == "__main__":
    # # Loop over rvs params
    # for param in params_tnorm_rvs:
    #     test_tnorm_rvs(param)
    # test_tnorm_cdf()
    # test_tnorm_ppf()
    # test_tnorm_CI()

    for param in params_tnorm_rvs:
        test_dmu(param)
    
    print('~~~ The test_dists.py script worked successfully ~~~')