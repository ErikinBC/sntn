""""
Checks that the _solvers.conf_inf_solver actually gets the correct unitroots

python3 -m tests.test_conf_inf_solver
python3 -m pytest tests/test_conf_inf_solver.py -s
"""

# External
import numpy as np
from scipy.stats import norm, binom
from parameters import seed
# Internal
from sntn.utilities.utils import is_equal, check_err_cdf_tol, str2list
from sntn._solvers import conf_inf_solver, valid_approaches, di_default_methods, _process_args_kwargs_flatten

# Parameter and function set up
alpha = 0.05
c_alpha = norm.ppf(1-alpha/2)
eps = 1e-6
tol = 1e-2


##########################
# ---- (2) BINOMIAL  --- #

def test_binomial(n:int=50, p0:float=0.5, nsim:int=850, alpha:float=0.1):
    """Make sure we can get the 'exact' CIs from the binomial distribution using root finding"""
    # Generate data
    dist_h0 = binom(p=p0, n=n)
    n_obs = dist_h0.rvs(nsim, random_state=seed)
    p_obs = n_obs / n

    # Naive approach
    p_ci_q = np.c_[binom(p=p_obs, n=n).ppf(alpha/2) / n,
                binom(p=p_obs, n=n).ppf(1-alpha/2) / n]
    # We get a coverage which is much larger than expected
    cover_q = np.mean((p_ci_q[:,0] <= p0) & (p0 <= p_ci_q[:,1]))
    # Calculate the p-value expected a 100(1-alpha)% coverage....
    dist_cover = binom(p=1-alpha, n=nsim)
    pval_cover = dist_cover.cdf(nsim*cover_q)
    pval_cover = 2*min(pval_cover, 1-pval_cover)  # two-sided
    assert pval_cover <= alpha, 'expected to reject null'


    # Use the conf_inf_solver instead
    find_ci = conf_inf_solver(dist=binom, param_theta='p', alpha=alpha, verbose=True)
    # Because binom uses .cdf(p*n), we need to convert the p's to number of actual of realiziation
    def fun_x0(x):
        return x / n * 0.9 

    def fun_x1(x):
        return min(x / n * 1.1, 1-1e-3)

    p_ci_root = find_ci._conf_int(x=n_obs,approach='root_scalar',di_dist_args={'n':n},di_scipy={'method':'secant'}, mu_lb=1e-3, mu_ub=1-1e-3, fun_x0=fun_x0, fun_x1=fun_x1)
    cover_root = np.mean((p_ci_root[:,0] <= p0) & (p0 <= p_ci_root[:,1]))
    pval_root = dist_cover.cdf(nsim*cover_root)
    pval_root = 2*min(pval_root, 1-pval_root)  # two-sided
    assert pval_root > alpha, 'expected to NOT reject null'



###############################
# ---- (1A) GAUSSIAN (MU) --- #

# Derivative of CDF w.r.t. mean
def dPhi_dmu(loc:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
    """Calculate the derivative of the Gaussian CDF w.r.t. mu"""
    flatten, kwargs = _process_args_kwargs_flatten(args, kwargs)
    assert 'scale' in kwargs, 'the named parameter "scale" must be given to dPhi_dmu'
    z = (x-loc)/kwargs['scale']
    deriv = -norm.pdf(z) / kwargs['scale']
    if flatten:
        deriv = np.diag(deriv.flatten())
    return deriv


def test_gaussian_mu() -> None:
    """Checks that the analytical CIs based on quantile align with root finding to a high tolerance"""
    # Check derivative function
    x, mu, sd = 1, 3, 3
    dmu_ana = dPhi_dmu(mu, x, alpha, 'scale', sd)
    dmu_num = (norm(mu+eps, sd).cdf(x)-norm(mu-eps, sd).cdf(x))/(2*eps)
    is_equal(dmu_ana, dmu_num)

    # Generate data
    mu = [-1, 0, 1]
    sd = np.array([1, 2, 3])
    dist0 = norm(loc=mu, scale=sd)
    x = np.squeeze(dist0.rvs([1,len(mu)],random_state=seed))

    # Get the CIs (known formula)
    solver = conf_inf_solver(dist=norm, param_theta='loc', dF_dtheta=dPhi_dmu)
    ci_lb0, ci_ub0 = x-sd*c_alpha, x+sd*c_alpha
    dist_kwargs = {'scale':sd}

    # Check that error functions recognize this as the "true" solution
    check_err_cdf_tol(solver=solver, theta=ci_ub0, x=x, alpha=alpha/2, **dist_kwargs)

    # Check that the solver._conf_int method gets the same results
    def fun_x0(x):
        return x * 1.0 
    def fun_x1(x):
        return x * 1.01
    di_dist_args = {'scale':sd}
    di_scipy = {}
    mu_lb, mu_ub = -10, +10
    for approach in valid_approaches:
        methods = di_default_methods[approach]
        for method in methods:
            di_scipy['method'] = method
            ci_root = solver._conf_int(x=x, approach=approach, di_dist_args=di_dist_args, di_scipy=di_scipy, mu_lb=mu_lb, mu_ub=mu_ub, fun_x0=fun_x0, fun_x1=fun_x1)
            is_equal(ci_root[:,0], ci_lb0, tol)
            is_equal(ci_root[:,1], ci_ub0, tol)
        print(f'Testing was successfull for approach {approach}')


###############################
# ---- (1B) GAUSSIAN (SD) --- #

def dPhi_dsig(scale:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
    """Calculate the derivative of the Gaussian CDF w.r.t. mu"""
    # Process args
    flatten, kwargs = _process_args_kwargs_flatten(args, kwargs)
    assert 'loc' in kwargs, 'the named parameter "loc" must be given to dPhi_dsig'
    z = (x-kwargs['loc'])/scale
    deriv = -z * norm.pdf(z) / scale
    if flatten:
        deriv = np.diag(deriv.flatten())
    return deriv


def test_gaussian_sig() -> None:
    """Since increasing sigma will always """
    # Check derivative function
    x, mu, sd = 1, 3, 3
    dmu_ana = dPhi_dsig(sd, x, alpha, 'loc', mu)
    dmu_num = (norm(mu, sd+eps).cdf(x)-norm(mu, sd-eps).cdf(x))/(2*eps)
    is_equal(dmu_ana, dmu_num)

# Note that CIs are not possible, since when x-mu>0, then increasing sigma simply decreases the CDF away from 1 to a low of 0.5, since Phi(0)=0.5, whereas when x-mu<0, then increasing sigma increases the CDF away from 0 to a high of 0.5 for the same reason
# [norm(loc=0,scale=s).cdf(1) for s in np.linspace(1e-5, 10, 20)]
# [norm(loc=2,scale=s).cdf(1) for s in np.linspace(1e-5, 10, 20)]


if __name__ == "__main__":
    test_gaussian_mu()
    test_gaussian_sig()
    test_binomial()

    print('~~~ The test_conf_inf_solver.py script worked successfully ~~~')