"""
Make sure SNTN works as expected

python3 -m tests.test_dists_nts
python3 -m pytest tests/test_dists_nts.py -s
"""

# Internal
import pytest
import numpy as np
import pandas as pd
from time import time
from scipy.stats import binom
# External
from sntn.dists import nts
from parameters import seed
from sntn.utilities.utils import try_except_breakpoint

# Used for pytest
params_shape = [((1,)), ((5, )), ((3, 2)), ((2, 2, 2)),]
params_alpha = [ (0.2), (0.1), (0.05) ]

def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates NTS parameters"""
    np.random.seed(seed)
    mu1 = np.random.randn(*shape)
    mu2 = np.random.randn(*shape)
    tau21 = np.exp(np.random.randn(*shape)) + 1
    tau22 = np.exp(np.random.randn(*shape)) + 1
    a = mu2 - np.abs(np.random.randn(*shape)) - 0.1
    b = mu2 + np.abs(np.random.rand(*shape)) + 0.1
    c1 = np.random.rand(*shape)
    c2 = 1 - c1
    return mu1, tau21, mu2, tau22, a, b, c1, c2

@pytest.mark.parametrize("shape", params_shape)
def test_nts_ppf(shape:tuple or list, ndraw:int=1000000, tol_err:float=2e-2) -> None:
    """
    Checks that the quantile function works as expected:
    i) Do empirical quantiles of rvs align with ppf?
    """
    # shape, alpha, ndraw = params_shape[0], params_alpha[1], 1000
    # Percentiles to check
    p_seq = np.arange(0.01,1,0.01)
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b)
    x = dist.rvs(ndraw, seed)
    # Get the empirical quantile
    emp_q = np.quantile(x, p_seq, axis=0)
    # Get the theoretical quantile
    theory_q = dist.ppf(p_seq, method='root_loop', verbose=True, verbose_iter=50)
    # Compare the errors
    maerr = np.abs(emp_q - theory_q).max()
    assert maerr < tol_err, f'Maximum error {maerr} is greater than tolerance {tol_err} for shape={str(shape)}'



@pytest.mark.parametrize("shape", params_shape)
@pytest.mark.parametrize("alpha", params_alpha)
# @pytest.mark.parametrize("ndraw", [(25),(50),(250)])
def test_nts_conf_int(shape:tuple, alpha:float, ndraw:int=250) -> None:
    """
    Checks that the confidence intervals cover the true parameter at the expected rate
    
    Parameters
    ----------
    shape:              The dimensions of the underlying parameters to take

    """
    # Distribution to check p-value for coverage...
    # shape, alpha, ndraw = params_shape[0], params_alpha[1], 100
    print('Drawing data')
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist_coverage = binom(n=int(ndraw*np.prod(shape)), p=1-alpha)

    # (ii) Repeat for fixed means
    np.random.seed(seed)
    idx_mu1 = np.random.rand(*mu1.shape) < 0.5
    mu = np.where(idx_mu1, mu1, mu2)
    dist = nts(mu, tau21, None, tau22, a, b, fix_mu=True)
    x = np.squeeze(dist.rvs(ndraw, seed))
    stime = time()
    print('Finding CIs')
    param_ci = dist.conf_int(x=x, alpha=alpha, param_fixed='mu', approach='root', verbose=True, verbose_iter=50)
    dtime, nroot = time() - stime, int(np.prod(param_ci.shape))
    rate = nroot / dtime
    print(f'Calculate {rate:.1f} roots per second (seconds={dtime:.0f}, roots={nroot})')
    ci_lb, ci_ub = np.take(param_ci, 0, -1), np.take(param_ci, 1, -1)
    # Have ground truth align with dimensions
    bcast_shape = np.broadcast_shapes(mu1.shape, ci_lb.shape)
    val_gt = np.broadcast_to(mu, bcast_shape)
    mu_gt = np.broadcast_to(dist.mean(), bcast_shape)
    # Assign values flat
    tmp_df = pd.DataFrame({'param':'mu', 'x':x.flat, 'lb':ci_lb.flat, 'ub':ci_ub.flat, 'gt':val_gt.flat, 'x_mu':mu_gt.flat})
    # Make sure the means aligned
    print(tmp_df.groupby('gt')[['x','x_mu']].mean().round(2))
    # try_except_breakpoint((tmp_df.groupby('gt')[['x','x_mu']].mean().diff(axis=1)['x_mu'].abs() < 1).all(), f'Means were not within 0.1 of each other')
    tmp_df = tmp_df.assign(cover=lambda x: (x['lb'] <= x['gt']) & (x['ub'] >= x['gt']))
    print(tmp_df.groupby('gt')['cover'].mean().round(2))
    tmp_df = tmp_df.sort_values('x').reset_index(drop=True)
    pval_mu = dist_coverage.cdf(tmp_df['cover'].sum())
    pval_mu = 2*min(pval_mu, 1-pval_mu)
    cover_mu = tmp_df['cover'].mean()
    # try_except_breakpoint(pval_mu > 0.05, 'Coverage did not match expected level')
    print(f'target={1-alpha}, shape={str(shape)}, coverage={100*cover_mu:.1f}%, pval={100*pval_mu:.1f}%')
    
    # # Find the actual alpha/2, 1-alpha/2 quantile values, and confirm that coverage fails ONLY outside them
    # from scipy.optimize import root
    # x_lb = root(fun=lambda w, mu, tau21, tau22, a, b, alpha: nts(mu, tau21, None, tau22, a, b, fix_mu=True).cdf(w)-alpha/2, x0=1,args=(mu, tau21, tau22, a, b, alpha)).x[0]
    # x_ub = root(fun=lambda w, mu, tau21, tau22, a, b, alpha: nts(mu, tau21, None, tau22, a, b, fix_mu=True).cdf(w)-1 + alpha/2, x0=1,args=(mu, tau21, tau22, a, b, alpha)).x[0]
    # assert not tmp_df.query('x < @x_lb').cover.any()
    # assert not tmp_df.query('x > @x_ub').cover.any()
    # assert tmp_df.query('(x <= @x_ub) & (x >= @x_lb)').cover.all()

    # # (i) For different means
    # dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2, fix_mu=False)
    # x = dist.rvs(ndraw)
    # # Generate CI for mu{12}
    # holder_param = []
    # for param_name in ['mu1', 'mu2']:
    #     param_ci = dist.conf_int(x=x, alpha=alpha, param_fixed=param_name)
    #     ci_lb, ci_ub = np.take(param_ci, 0, -1), np.take(param_ci, 1, -1)
    #     # Broadcast...
    #     val_gt = getattr(dist, param_name)
    #     val_gt = np.broadcast_to(val_gt, np.broadcast_shapes(val_gt.shape, ci_lb.shape))
    #     tmp_df = pd.DataFrame({'param':param_name, 'x':np.squeeze(x), 'lb':ci_lb, 'ub':ci_ub, 'gt':val_gt})
    #     holder_param.append(tmp_df)
    # res_param = pd.concat(holder_param).reset_index(drop=True)
    



@pytest.mark.parametrize("shape", params_shape)
def test_nts_cdf(shape:tuple, ndraw:int=10000, tol_cdf:float=0.01) -> None:
    """Checks that:
    i) CDF aligns with classic 1964 paper
    ii) Empirical rvs aligns with cdf
    """ 
    # (i) Sanity check on 1964 query
    mu1, tau21 = 100, 6**2
    mu2, tau22 = 50, 3**2
    a, b = 44, np.inf
    w = 138
    dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
    expected_1964 = 0.03276
    cdf_1964 = dist_1964.cdf(w)[0]
    assert np.round(cdf_1964,5) == expected_1964, F'Expected CDF to be: {expected_1964} not {cdf_1964}' 
    
    # (ii) Check that random parameters align with rvs
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    x = dist.rvs(1)[0,...]
    cdf_theory = dist.cdf(x)
    cdf_rvs = np.mean(dist.rvs(ndraw, seed) <= x, 0)
    err_max = np.abs(cdf_theory - cdf_rvs).max()
    assert err_max < tol_cdf, f'Maximum error {err_max} was greater than {tol_cdf}'

    

@pytest.mark.parametrize("shape", params_shape)
def test_nts_pdf(shape:tuple, tol_cdf:float=0.005, tol_mu:float=0.1) -> None:
    """Checks that:
    i) CDF integrates to one
    ii) Mean = int f(x) x dx 
    iii) Median = inf_c : int_{lb}^c f(x) dx = 0.5
    """ 
    # Draw different distributions
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    # Check that numerical integration matches theory for mean
    data = dist.rvs(10000, seed)
    dmin = np.min(data, 0) - 1
    dmax = np.max(data, 0) + 1
    points = np.linspace(dmin, dmax, 25000)
    dpoints = np.expand_dims((points[1] - points[0]).reshape(shape), 0)    
    # The pdf should integrate to close to one
    f = dist.pdf(points)
    probs = f * dpoints
    total_prob = np.sum(probs, 0)
    assert np.all(np.abs(total_prob - 1) < tol_cdf), 'Integral should sum to 1!'
    # Mean should match...
    mu_int = np.sum(f * points * dpoints, 0)
    assert np.all(np.abs(mu_int - np.mean(data, 0)) < tol_mu), 'Means do not align'
    # Median should match...
    idx_median = np.expand_dims(np.argmin(np.cumsum(probs, 0) <= 0.5,axis=0),0)
    med_int = np.take_along_axis(points, idx_median, 0)
    med_data = np.median(data, 0, keepdims=True)
    abs_err_med = np.abs(med_int - med_data)
    assert np.all(abs_err_med < tol_mu), 'Medians do not align'


@pytest.mark.parametrize("shape", params_shape)
def test_nts_rvs(shape:tuple, nsim:int=10000, tol:float=1e-1) -> None:
    """Checks that the random variables align with expected mean"""
    # Draw different distributions
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    data = dist.rvs(nsim, seed)
    # Check that the mean of the rvs aligns with E[]
    mu_rvs = data.mean(0)
    mu_theory = dist.mean()
    assert np.all(np.abs(mu_rvs - mu_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'
    # # Check median
    # med_rvs = np.median(data, 0)
    # med_theory = dist.ppf(0.5)
    # assert np.all(np.abs(med_rvs - med_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'
    


if __name__ == "__main__":
    print('--- test_nts_rvs ---')
    test_nts_rvs()
    test_nts_pdf()


    print('~~~ The test_dists_nts.py script worked successfully ~~~')