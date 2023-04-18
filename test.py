import numpy as np
from scipy.stats import norm
from parameters import seed
from sntn.utilities.utils import is_equal
from sntn._solvers import conf_inf_solver

# Parameter and function set up
alpha = 0.05
c_alpha = norm.ppf(1-alpha/2)

def dPhi_dmu(x, alpha, loc, scale):
    """Calculate the derivative of the Gaussian CDF w.r.t. mu"""
    z = (x-loc)/scale
    deriv = -norm.pdf(z) / scale
    return deriv

# Check derivative function
eps = 1e-6
x, mu, sd = 1, 3, 3
dmu_ana = dPhi_dmu(x, mu, sd)
dmu_num = (norm(mu+eps, sd).cdf(x)-norm(mu-eps, sd).cdf(x))/(2*eps)
is_equal(dmu_ana, dmu_num)


# Generate data
mu = [-1, 0, 1]
sd = np.array([1, 2, 3])
dist0 = norm(loc=mu, scale=sd)
x = np.squeeze(dist0.rvs([1,len(mu)],random_state=seed))
# Get the CIs
solver = conf_inf_solver(norm, param_theta='loc', dF_dtheta=dPhi_dmu)
ci_ub0 = x+sd*c_alpha
# Check that the "error" is zero at the true solution
di0 = {'loc':ci_ub0,'scale':sd}
assert np.all(solver._err_cdf(x, alpha/2, **di0) == 0)
assert solver._err_cdf0(x[0], alpha/2, loc==ci_ub0[0], scale=sd[0]) == 0
assert np.all(solver._err_cdf2(x, alpha/2, **di0) == 0)
solver._derr_cdf2(x, alpha/2, **di0)


print(x)
