""""
Checks that the _solvers.conf_inf_solver actually gets the correct unitroots

python3 -m tests.test_conf_inf_solver
python3 -m pytest tests/test_conf_inf_solver.py -s
"""

import numpy as np
from scipy.stats import norm
from parameters import seed
from sntn.utilities.utils import is_equal
from sntn._solvers import conf_inf_solver

# Parameter and function set up
alpha = 0.05
c_alpha = norm.ppf(1-alpha/2)
eps = 1e-6
tol = 1e-7


def check_err_cdf_tol(solver:conf_inf_solver, theta:np.ndarray, x:np.ndarray, alpha:float, **dist_kwargs) -> None:
    """Make sure that the root/squared error is zero and the correct solution
    
    Parameters
    ----------
    solver:             The constructed conf_inf_solver class
    theta:              Candidate CI value (i.e. upper or lower bound)
    x:                  Observed statistic
    alpha:              Type-I error (i.e. alpha/2)
    dist_kwargs:        Arguments to pass into solver._{err_cdf,err_cdf0,err_cdf2,derr_cdf2}
    """
    assert isinstance(solver, conf_inf_solver), 'solver needs to be of type conf_inf_solver'
    has_dF_dtheta = hasattr(solver, 'dF_dtheta')
    n = len(x)
    nudge = 0.1
    # Going to broadcast alpha for looping
    _, alpha = np.broadcast_arrays(theta, alpha)
    # Check that the "error" is zero at the true solution
    di_eval = {**{'theta':theta.copy(), 'x':x, 'alpha':alpha}, **dist_kwargs}
    assert np.all(solver._err_cdf(**di_eval) == 0), 'Root was not zero'
    assert np.all(solver._err_cdf2(**di_eval) == 0), 'Squared-error was not zero'
    for i in range(n):
        di_eval_i = {k:v[i] for k,v in di_eval.items()}
        assert solver._err_cdf0(**di_eval_i) == 0, 'Root (float) was not zero'
    if has_dF_dtheta:
        assert np.all(solver._derr_cdf2(**di_eval) == 0), 'Derivative was not zero'
    # Check the error is non-zero at a permuted distribution
    di_eval['theta'] += nudge
    assert np.all(solver._err_cdf(**di_eval) != 0), 'Root was zero'
    assert np.all(solver._err_cdf2(**di_eval) != 0), 'Squared-error was zero'
    if has_dF_dtheta:
        assert np.all(solver._derr_cdf2(**di_eval) != 0), 'Derivative was zero'


def test_gaussian_mu() -> None:
    None

# Derivative of CDF w.r.t. mean
def dPhi_dmu(loc:np.ndarray, x:np.ndarray, alpha:float, scale:np.ndarray) -> np.ndarray:
    """Calculate the derivative of the Gaussian CDF w.r.t. mu"""
    z = (x-loc)/scale
    deriv = -norm.pdf(z) / scale
    return deriv

# Check derivative function
x, mu, sd = 1, 3, 3
dmu_ana = dPhi_dmu(mu, x, alpha, sd)
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
di_scipy = {'method':'secant'}
mu_lb, mu_ub = -10, +10
ci_root = solver._conf_int(x=x, approach='root_scalar', di_dist_args=di_dist_args, di_scipy=di_scipy, mu_lb=-10, mu_ub=+10)
is_equal(ci_root[:,0], ci_lb0, tol)
is_equal(ci_root[:,1], ci_ub0, tol)


#########################
# ---- (2) ... --- #


if __name__ == "__main__":
    test_gaussian_mu()

    print('~~~ The test_conf_inf_solver.py script worked successfully ~~~')