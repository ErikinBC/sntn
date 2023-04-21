""""
Checks that the _solvers.conf_inf_solver actually gets the correct unitroots

python3 -m tests.test_conf_inf_solver
python3 -m pytest tests/test_conf_inf_solver.py -s
"""

# External
import numpy as np
from scipy.stats import norm
from parameters import seed
# Internal
from sntn._solvers import conf_inf_solver, valid_approaches, di_default_methods
from sntn.utilities.utils import is_equal, check_err_cdf_tol, str2list

# Parameter and function set up
alpha = 0.05
c_alpha = norm.ppf(1-alpha/2)
eps = 1e-6
tol = 1e-2


#########################
# ---- (1) GAUSSIAN --- #

# Derivative of CDF w.r.t. mean
def dPhi_dmu(loc:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
    """Calculate the derivative of the Gaussian CDF w.r.t. mu"""
    if len(args) > 0:
        di_names = str2list(args[0])
        if len(di_names) > 1:
            kwargs = dict(zip(di_names, args[1:][0]))
        else:
            kwargs = dict(zip(di_names, args[1:]))
    assert 'scale' in kwargs, 'the named parameter "scale" must be given to dPhi_dmu'
    flatten, kwargs = conf_inf_solver._check_flatten(**kwargs)
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
    di_dist_args = {'scale':sd}
    di_scipy = {}
    mu_lb, mu_ub = -10, +10
    for approach in valid_approaches:
        methods = di_default_methods[approach]
        for method in methods:
            di_scipy['method'] = method
            ci_root = solver._conf_int(x=x, approach=approach, di_dist_args=di_dist_args, di_scipy=di_scipy, mu_lb=-10, mu_ub=+10)
            is_equal(ci_root[:,0], ci_lb0, tol)
            is_equal(ci_root[:,1], ci_ub0, tol)
        print(f'Testing was successfull for approach {approach}')


############################
# ---- (2) EXPONENTIAL --- #


#########################
# ---- (3) BINOMIAL --- #


##########################
# ---- (4) TRUNCNORM --- #



if __name__ == "__main__":
    test_gaussian_mu()

    print('~~~ The test_conf_inf_solver.py script worked successfully ~~~')