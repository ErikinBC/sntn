"""
Make sure SNTN works as expected

python3 -m tests.test_dists_nts
python3 -m pytest tests/test_dists_nts.py -s
python3 -m pytest tests/test_dists_nts.py -s -k 'test_nts_ppf'
"""

# Internal
import pytest
import numpy as np
import pandas as pd
from time import time
from scipy.stats import binom, uniform
# External
from sntn.dists import nts
from parameters import seed
from sntn.utilities.utils import try_except_breakpoint

# Used for pytest
params_shape = [((1,)), ((5, )), ((3, 2)), ((2, 2, 2)),]
params_alpha = [ (0.2), (0.1), (0.05) ]
params_ppf_method = [ 'fast', 'loop', 'root', ]

def gen_params(shape:tuple | list, seed:int | None) -> tuple:
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
  

def test_1964():
    """
    Checks that answer aligns with classic 1964 paper:
    Query 2: The Sum of Values from a Normal and a Truncated Normal Distribution (Continued)
    https://www.jstor.org/stable/1266101?seq=1
    """
    # Classic parameters
    mu1, tau21 = 100, 6**2
    mu2, tau22 = 50, 3**2
    a, b = 44, np.inf
    w = 138
    dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
    p_seq = np.arange(0.05, 1, 0.05)
    # Generate the naive quantiles
    quant_loop = np.squeeze(dist_1964.ppf(p_seq, method='loop'))
    quant_root = np.squeeze(dist_1964.ppf(p_seq, method='root'))
    quant_fast = np.squeeze(dist_1964.ppf(p_seq, method='fast'))
    np.testing.assert_allclose(quant_loop, quant_root)
    np.testing.assert_allclose(quant_loop, quant_fast)
    # Ensure the CDF aligns
    cdf_loop = np.squeeze(dist_1964.cdf(quant_loop, method='bvn'))
    cdf_fast = np.squeeze(dist_1964.cdf(quant_fast, method='fast'))
    np.testing.assert_allclose(cdf_loop, p_seq)
    np.testing.assert_allclose(cdf_fast, p_seq)


def test_runtime():
    """
    # Check that we can clock >10k roots per second w/ infinity in the alpha/beta limits
    """    
    pct_infty = 0.05
    nvecs = [10000, 25000, 50000, 100000, ]
    holder = []
    for nvec in nvecs:
        print(f'Vector size = {nvec}')
        # break
        mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(((nvec,)), seed=seed)
        # Through in some infinities
        a = np.where(uniform.rvs(size=nvec, random_state=seed) < pct_infty, -np.infty, a)
        b = np.where(uniform.rvs(size=nvec, random_state=seed+1) < pct_infty, +np.infty, b)
        dist_sntn = nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b, c1=c1, c2=c2)
        alphas = uniform.rvs(size=nvec, random_state=seed)
        stime = time()
        quant = np.squeeze(dist_sntn.ppf(p=alphas, method='fast'))
        dtime = time() - stime
        failed_quants = np.isnan(quant)
        print(f'Number of failed quantiles = {failed_quants.sum()}')
        np.testing.assert_allclose(dist_sntn.cdf(quant), alphas)
        holder.append([nvec, dtime])
    # Merge and show
    res_runtime = pd.DataFrame(holder, columns = ['n', 'time']).assign(rate=lambda x: (x['n']/x['time']).astype(int))
    print('Quantiles per second using "fast" method')
    print(res_runtime)


@pytest.mark.parametrize("shape", params_shape)
@pytest.mark.parametrize("ppf_method", params_ppf_method)
def test_nts_ppf(shape:tuple | list, ppf_method:str, ndraw:int=2000000, tol_err:float=0.01) -> None:
    """
    Checks that the quantile function works as expected by comparing it to the empirical quantile from sampling from the distribution (which is trivial to do)
    """
    # Percentiles to check
    p_seq = np.arange(0.01, 1, 0.01)
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    x = dist.rvs(ndraw, seed)
    # Get the empirical quantile
    emp_q = np.quantile(x, p_seq, axis=0)
    # Get the theoretical quantile
    stime = time()
    theory_q = dist.ppf(p_seq, method=ppf_method)
    dtime = time() - stime
    # Compare the errors
    maerr = np.abs(emp_q - theory_q).max()
    print(f'\nMax error: {maerr:.4f} (method = {ppf_method}), time={dtime:.3f} seconds')
    try:
        assert maerr < tol_err, f'Maximum error {maerr} is greater than tolerance {tol_err} for shape={str(shape)}'
    except:
        breakpoint()


@pytest.mark.parametrize("shape", params_shape)
def test_nts_cdf(shape:tuple, ndraw:int=20000, tol_cdf:float=0.01) -> None:
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
    cdf_1964 = dist_1964.cdf(w, method='bvn')[0]
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
    # Check median
    med_rvs = np.median(data, 0)
    med_theory = dist.ppf(p=0.5)
    assert np.all(np.abs(med_rvs - med_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'


@pytest.mark.parametrize("shape", params_shape)
@pytest.mark.parametrize("alpha", params_alpha)
def test_nts_conf_int(shape:tuple, alpha:float, ndraw:int=250, tol_type1:float=0.01, tol_xmu:float | None=0.3, verbose_iter:int=5, n_chunks:int=10, cdf_approach:int='scipy') -> None:
    """
    Checks thats:
    i) conf_int method does not error out
    ii) Coverage is at expected levels (tol_type1)
    iii) Average of ndraws is close to mean theory (tol_xmu)
    iv) When the CI does not cover the true parameter, the p-value should be <alpha/2 or >1-alpha/2
    v) When the CI does not cover the true parameter, the value of x is greater than critical value (using ppf)
    
    Parameters
    ----------
    shape:              The dimensions of the underlying parameters to assume (the number of NTS distributions)
    alpha:              The desired type-I error
    ndraw:              For each NTS distribution, how many samples (and hence CIs) to draw from?
    tol_type1:          Assuming (1-alpha)% coverage of binomial null dist, pvalue needed before rejecting null
    tol_xmu:            The average of draw values should be within tol_xmu of the theoretical mean
    verbose_iter:       For each CI, after how many solutions should the status be printed?
    n_chunks:           ..
    cdf_approach:       Which BVN method should be used (default='scipy')



    """
    # Draw data
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist_coverage = binom(n=int(ndraw*np.prod(shape)), p=1-alpha)

    # Assume fixed means
    np.random.seed(seed)
    idx_mu1 = np.random.rand(*mu1.shape) < 0.5
    mu = np.where(idx_mu1, mu1, mu2)
    dist_gt = nts(mu, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='owen')
    x = np.squeeze(dist_gt.rvs(ndraw, seed))
    cdf_gt = dist_gt.cdf(x)
    crit_lb = dist_gt.ppf(alpha/2)
    crit_ub = dist_gt.ppf(1-alpha/2)

    # Find the CIs
    print(f'~~~ Finding CIs for target={1-alpha}, shape={str(shape)} ~~~')
    stime = time()
    dist_ci = nts(1, tau21, None, tau22, a, b, fix_mu=True)
    param_ci = dist_ci.conf_int(x=x, alpha=alpha, param_fixed='mu', approach='root', verbose=True, verbose_iter=verbose_iter, cdf_approach=cdf_approach, n_chunks=n_chunks)
    dtime, nroot = time() - stime, int(np.prod(param_ci.shape))
    rate = nroot / dtime
    print(f'Calculated {rate:.1f} roots per second (seconds={dtime:.0f}, roots={nroot})')
    ci_lb, ci_ub = np.take(param_ci, 0, -1), np.take(param_ci, 1, -1)
    
    # Compare CIs to ground truth
    bcast_shape = np.broadcast_shapes(mu1.shape, ci_lb.shape)
    val_gt = np.broadcast_to(mu, bcast_shape)
    mu_gt = np.broadcast_to(dist_gt.mean(), bcast_shape)
    crit_lb = np.broadcast_to(crit_lb, bcast_shape)
    crit_ub = np.broadcast_to(crit_ub, bcast_shape)
    # Get the CDF of the x-values
    # Assign values flat
    tmp_df = pd.DataFrame({'param':'mu', 'x':x.flat, 'cdf':cdf_gt.flat, 'crit_lb':crit_lb.flat, 'crit_ub':crit_ub.flat, 'lb':ci_lb.flat, 'ub':ci_ub.flat, 'gt':val_gt.flat, 'x_mu':mu_gt.flat})
    # Make sure the means aligned
    if tol_xmu is not None:
        try_except_breakpoint((tmp_df.groupby('gt')[['x','x_mu']].mean().diff(axis=1)['x_mu'].abs() < tol_xmu).all(), f'Means were not within {tol_xmu} of each other')
    
    # Calculate the coverage probability
    tmp_df = tmp_df.assign(cover=lambda x: (x['lb'] <= x['gt']) & (x['ub'] >= x['gt']))
    # print(tmp_df.groupby('gt')['cover'].mean().round(2).reset_index().assign(target=1-alpha))
    # tmp_df = tmp_df.sort_values('x').reset_index(drop=True)
    pval_mu = dist_coverage.cdf(tmp_df['cover'].sum())
    pval_mu = 2*min(pval_mu, 1-pval_mu)
    cover_mu = tmp_df['cover'].mean()
    try_except_breakpoint(pval_mu > tol_type1, f'Prob of coverage being expected level was less than {tol_type1}')
    print(f'~~~ coverage={100*cover_mu:.1f}%, pval={100*pval_mu:.1f}% ~~~')
    
    # Ensure that the areas that fail coverage align with expected quantiles of x (due to monotonicty)
    cdf_gt_pos = tmp_df.loc[tmp_df['cover'],'cdf']
    cdf_gt_neg = tmp_df.loc[~tmp_df['cover'],'cdf']
    mx_pval_neg = np.max(np.minimum(1-cdf_gt_neg, cdf_gt_neg))
    assert mx_pval_neg < alpha/2, f'Expected max GT pvalue to be less than {alpha/2}: {mx_pval_neg}'
    mi_pval_pos = np.min(cdf_gt_pos)
    mx_pval_pos = np.max(cdf_gt_pos)
    # assert mi_pval_pos >= alpha/2, f'Minimum p-values for covered params should be at least {alpha/2}: {mi_pval_neg}'
    assert mx_pval_pos <= 1-alpha/2, f'Maximum p-values for covered params should be at most {1-alpha/2}: {mx_pval_neg}'

    # Repeat for the "critical" value (should have 1:1 mapping with p-values)    
    cn_crit = ['cover','x','crit_lb','crit_ub']
    tmp_crit = tmp_df[cn_crit].assign(reject=lambda z: (z['x'] < z['crit_lb']) | (z['x'] > z['crit_ub']))
    tmp_crit = tmp_crit.assign(check=lambda x: x['cover'] != x['reject'])
    assert tmp_crit['check'].all(), 'Expected critical values to be aligned with coverage to be aligned with p-values'

    # # Repeat experiments for different means
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

if __name__ == "__main__":
    shape_test = params_shape[2]
    alpha_test = params_alpha[1]
    
    print('--- test_nts_rvs ---')
    test_nts_rvs(shape=shape_test)
    
    print('--- test_nts_cdf ---')
    test_nts_cdf(shape=shape_test)
    
    print('--- test_nts_pdf ---')
    test_nts_pdf(shape=shape_test)
    
    print('--- test_nts_ppf ---')
    test_nts_ppf(shape=shape_test, ppf_method='fast')

    print('--- test_nts_conf_int ---')
    # Try 1000 random parameterizations with a single draw (contrasted to 250 draws for a handful of parameters)
    # Try Owen's method to make sure that scipy backup works as expected
    # test_nts_conf_int(shape=(20,10,5), alpha=alpha_test, ndraw=1, tol_xmu=None, verbose_iter=1, n_chunks=1, cdf_approach='owen')

    print('~~~ The test_dists_nts.py script worked successfully ~~~')