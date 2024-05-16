"""
Do experiments on ppf solving runtime

python3 -m simulations.6_ppf_runtime
"""

# External modules
import unittest
import numpy as np
import pandas as pd
from time import time
from timeit import timeit
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm, uniform, expon
from scipy.optimize import root_scalar, root, newton
# Internal modules
from sntn import dists
from sntn._bvn import _bvn as bvn
from sntn._fast_integrals import Phi_diff, bvn_cdf_diff, dbvn_cdf_diff, d2bvn_cdf_diff, _rootfinder_newton

# Set seed
seed = 1234

# Set up for assertion checks
case = unittest.TestCase()

# Data generation function
def _generate_random_sntn_params(size, seed):
    mu1 = norm.rvs(size=size, random_state=seed)
    mu2 = norm.rvs(size=size, random_state=seed+1)
    tau21 = expon(scale=1).rvs(size=size, random_state=seed) + 1
    tau22 = expon(scale=1).rvs(size=size, random_state=seed+1) + 1
    a = uniform(loc=-3, scale=1).rvs(size=size, random_state=seed)
    b = a + 4
    return mu1, mu2, tau21, tau22, a, b


# # Classic parameters
# mu1, tau21 = 100, 6**2
# mu2, tau22 = 50, 3**2
# a, b = 44, np.inf
# w = 138
# dist_1964 = dists.nts(mu1, tau21, mu2, tau22, a, b)
# p_seq = np.arange(0.05, 1, 0.05)
# # Generate the naive quantiles
# quant_approx = np.squeeze(dist_1964.ppf(p_seq, method='approx'))
# quant_loop = np.squeeze(dist_1964.ppf(p_seq, method='loop'))
# quant_root = np.squeeze(dist_1964.ppf(p_seq, method='root'))
# quant_fast = np.squeeze(dist_1964.ppf(p_seq, method='fast'))
# cdf_loop = np.squeeze(dist_1964.cdf(quant_loop, method='bvn'))
# cdf_fast = np.squeeze(dist_1964.cdf(quant_fast, method='fast'))
# np.testing.assert_allclose(cdf_loop, p_seq)
# np.testing.assert_allclose(cdf_fast, p_seq)



########################
# --- (0) RUNTIME! --- #

# Check that we can clock >10k roots per second
nvecs = [10000, 25000, 50000, 100000, 500000, 1000000]
holder = []
for nvec in nvecs:
    print(f'Vector size = {nvec}')
    # break
    mu1, mu2, tau21, tau22, a, b = _generate_random_sntn_params(nvec, seed)
    dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)
    alphas = uniform.rvs(size=nvec, random_state=seed)
    stime = time()
    quant = np.squeeze(dist_sntn.ppf(p=alphas, method='fast', clip=np.infty))
    dtime = time() - stime
    failed_quants = np.isnan(quant)
    print(f'Number of failed quantiles = {failed_quants.sum()}')
    # print(np.where()[0])
    # np.testing.assert_allclose(dist_sntn.cdf(quant), alphas)
    holder.append([nvec, dtime])
# Merge and show
res_runtime = pd.DataFrame(holder, columns = ['n', 'time']).assign(rate=lambda x: (x['n']/x['time']).astype(int))
print('Quantiles per second using "fast" method')
print(res_runtime)


import sys
sys.exit('stop here')

#################################
# --- (1) DIFFERENCE IN PHI --- #

# NOTE: THERE IS A SMALL OPPORTUNITY FOR A SPEED-UP IN CALCULATING THE DIFFERENCE IN NORMAL CDFs
tnum = 1000
nvec = 1122
ub = norm.rvs(size=nvec, random_state=seed)
ub[0] = np.infty
lb = ub + uniform(loc=-3, scale=1).rvs(size=nvec, random_state=seed)
cdf1 = norm.cdf(ub) - norm.cdf(lb)
cdf2 = Phi_diff(ub, lb)
np.testing.assert_allclose(cdf1, cdf2, atol=1e-15)
tdcdf = timeit("norm.cdf(ub) - norm.cdf(lb)", number=tnum, globals=globals())
tdPhi = timeit("Phi_diff(ub, lb)", number=tnum, globals=globals())

print(f'Time it takes for norm.cdf={tdcdf:.2f} vs dPhi={tdPhi:.2f} (n={nvec}, times={tnum})')


###########################
# --- (2) FASTEST CDF --- #

# Check the scalar
rho = 0.62
delta = 1.0
omega = -2.0
m = -0.2

bvn_owen = bvn(0, 1, 0, 1, rho, cdf_approach='owen')
q1 = bvn_owen.cdf(x1=m, x2=delta) - bvn_owen.cdf(x1=m, x2=omega)
q2 = bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho)
np.testing.assert_allclose(q2, q1)

# Check speed on vector
tnum = 1000
n = 105
rho = 4/5
x = norm.rvs(size=[n, 2], random_state=seed)
x1, x2 = x.T
bvn_scipy = bvn(0, 1, 0, 1, rho, cdf_approach='scipy')
bvn_owen = bvn(0, 1, 0, 1, rho, cdf_approach='owen')
bvn_mvn = mvn(mean=[0,0],cov=[[1,rho],[rho,1]])
cdf_scipy = bvn_scipy.cdf(x1=x1, x2=x2)
cdf_owen = bvn_owen.cdf(x1=x1, x2=x2)
cdf_mvn = bvn_mvn.cdf(x)
np.testing.assert_allclose(cdf_scipy, cdf_owen)
np.testing.assert_allclose(cdf_scipy, cdf_mvn)
# Default CDF
towen = timeit("bvn_owen.cdf(x1=x1, x2=x2)", globals=globals(), number=tnum)
tscipy = timeit("bvn_scipy.cdf(x1=x1, x2=x2)", globals=globals(), number=tnum)
tmvn = timeit("bvn_mvn.cdf(x)", globals=globals(), number=tnum)
print(f'Time it takes to run the normal BVN CDF {tnum} times: \nOwen={towen:.3f}, Scipy={tscipy:.3f}, and MVN={tmvn:.3f}')
# Difference in CDFs
tdiff = timeit("bvn_cdf_diff(x1=x1, x2a=delta, x2b=omega, rho=rho)", globals=globals(), number=tnum)
towen = timeit("bvn_owen.cdf(x1=x1, x2=delta) - bvn_owen.cdf(x1=x1, x2=omega)", globals=globals(), number=tnum)
print(f'Run-time for vector of length {n}, {tnum} times:\nbvn_cdf_diff={tdiff:0.2f}, bvn_owen={towen:0.2f}')


##########################
# --- (3) M-QUANTILE --- #

# note that in the SNTN CDF, m1(z) = (z - theta1)/sigma1
#       so z(m1) = sigma1*m1 + theta1

# Some parameters
nsim = 1257
alpha = 0.13

# Check that default PPF works as expected
mu1, mu2, tau21, tau22, a, b = _generate_random_sntn_params(nsim, seed)
dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)
quant_loop = np.atleast_1d(np.squeeze(dist_sntn.ppf(p=alpha, method='loop')))
quant_m_alt = _rootfinder_newton(ub=dist_sntn.beta, lb=dist_sntn.alpha, rho=dist_sntn.rho,
                                 target_p=dist_sntn.Z*alpha)
quant_loop_alt = quant_m_alt*dist_sntn.sigma1 + dist_sntn.theta1
df_p_alt = pd.DataFrame({'q':quant_loop, 'q2':quant_loop_alt})
idx_worst = df_p_alt.diff(axis=1)['q2'].abs().sort_values().tail(20).index
# df_p_alt.loc[idx_worst]
np.testing.assert_allclose(quant_loop, quant_loop_alt)
p_quant = np.atleast_1d(np.squeeze(dist_sntn.cdf(quant_loop)))
print(f'Calculated quantile={quant_loop[0]:.3f} for p-value ({alpha:.2f}), and cdf={p_quant[0]:.3f}')

# Check it can be recovered with bvn_cdf_diff
m_p = (quant_loop - dist_sntn.theta1) / dist_sntn.sigma1
target_p = dist_sntn.Z * alpha
np.testing.assert_allclose(bvn_cdf_diff(m_p, dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho) , target_p)

# -- Quick and dirty root finding -- #
# Get the baseline time
tym_ppf_bl = timeit("dist_sntn.ppf(p=alpha, method='loop')", globals=globals(), number=1)
nsim / tym_ppf_bl

# Check that rootfun finds the first quantile
rootfun = lambda z, ub, lb, rho, p: bvn_cdf_diff(z, ub, lb, rho) - p
drootfun = lambda z, ub, lb, rho, p: d2bvn_cdf_diff(z, ub, lb, rho)
def drootfun_clip(z, ub, lb, rho, clip_low:float=0.1, clip_high:float=1.1):
    grad = dbvn_cdf_diff(z, ub, lb, rho)
    grad_clip = np.sign(grad) * np.clip(np.abs(grad), clip_low, clip_high)
    return grad_clip
d2rootfun = lambda z, ub, lb, rho, p: d2bvn_cdf_diff(z, ub, lb, rho)

m_p_root = root_scalar(f=rootfun, args=(dist_sntn.beta[0], dist_sntn.alpha[0], dist_sntn.rho[0], target_p[0]), method='brentq', bracket=(-5, 5)).root
np.testing.assert_allclose(m_p[0], m_p_root)
quant_root = m_p_root*dist_sntn.sigma1[0] + dist_sntn.theta1[0]
np.testing.assert_allclose(quant_root, quant_loop[0])

# Run the solvers that only need a bound
solvers_bracket = ['brentq', 'brenth']
solvers_x0x1 = ['secant']
solvers_grad = ['newton']
solvers_hess = ['halley']
solvers = solvers_bracket + solvers_grad + solvers_hess  #  + solvers_x0x1
bracket = (-5, 5)
x0, x1 = -1, +1
holder_solver = []
for solver in solvers:
    print(f'solver = {solver}')
    roots = np.zeros(nsim)
    if solver in solvers_bracket:
        stime = time()
        for i in range(nsim):
            roots[i] = root_scalar(f=rootfun, args=(dist_sntn.beta[i], dist_sntn.alpha[i], dist_sntn.rho[i], target_p[i]), method=solver, bracket=bracket).root
    if solver in solvers_x0x1:
        stime = time()
        for i in range(nsim):
            roots[i] = root_scalar(f=rootfun, args=(dist_sntn.beta[i], dist_sntn.alpha[i], dist_sntn.rho[i], target_p[i]), method=solver, x0=x0, x1=x1).root
    if solver in solvers_grad:
        stime = time()
        for i in range(nsim):
            roots[i] = root_scalar(f=rootfun, args=(dist_sntn.beta[i], dist_sntn.alpha[i], dist_sntn.rho[i], target_p[i]), method=solver, x0=x0, fprime=drootfun_clip).root
    if solver in solvers_hess:
        stime = time()
        for i in range(nsim):
            roots[i] = root_scalar(f=rootfun, args=(dist_sntn.beta[i], dist_sntn.alpha[i], dist_sntn.rho[i], target_p[i]), method=solver, x0=x0, fprime=drootfun_clip, fprime2=d2rootfun).root
    dtime = time() - stime
    df_i = pd.DataFrame({'solver':solver, 'dtime':dtime, 'roots':roots})
    holder_solver.append(df_i)

# Run the solvers that need 
res_solvers = pd.concat(holder_solver).rename_axis('idx')
res_solvers.groupby('solver')['dtime'].max().reset_index().assign(rate=lambda x: nsim/x['dtime'])
comp_roots = res_solvers.reset_index().pivot(index='idx',columns='solver', values='roots')
comp_roots.corr()
# Repeat for the root-vectorized methods
x0_vec = np.repeat(-1, nsim)
roots_newton = newton(func=rootfun, x0=x0_vec, fprime=drootfun_clip, args=(dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho, target_p,))
roots_newton2 = newton(func=rootfun, x0=x0_vec, fprime=drootfun_clip, fprime2=d2rootfun, args=(dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho, target_p,))
roots_median = res_solvers.groupby('idx')['roots'].median()
roots_hybr = root(fun=rootfun, jac=lambda z, ub, lb, rho, p: np.diag(drootfun_clip(z, ub, lb, rho)), x0=x0_vec, args=(dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho, target_p,), method='hybr').x
# They achieve the same results, but the newton method is blazing fast!!
print(pd.DataFrame({'newton':roots_newton, 'newton2':roots_newton2, 'hybr':roots_hybr, 'med':roots_median}).corr())
# After transformation should be the same...
np.testing.assert_allclose(quant_loop, roots_newton*dist_sntn.sigma1 + dist_sntn.theta1)


###################################
# --- (4) NEWTON OPTIMZIATION --- #

# Set some quantile target
alpha = 0.189

# (1) GRADIENT CLIPPING IS NOT NEEDED WHEN THE HESSIAN IS IN PLAY! OTHERWISE, USE THE DEFAULT 0.1/5.0
nsim = 100000
tol = 1e-8

mu1, mu2, tau21, tau22, a, b = _generate_random_sntn_params(nsim, seed)
dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)

lbs = [0.0001, 0.001, 0.01, 0.1, 1]
ubs = [1, 2, 4, 8, 16]

holder = []
for lb in lbs:
    for ub in ubs:
        print(f'lb={lb}, ub={ub}')
        stime = time()
        roots, yval = _rootfinder_newton(ub=dist_sntn.beta, lb=dist_sntn.alpha, rho=dist_sntn.rho,
                                 target_p=dist_sntn.Z*alpha, clip_low=lb, clip_high=ub, ret_rootfun=True)
        dtime = time() - stime
        n_viol = (np.abs(yval) > tol).sum()
        holder.append([lb, ub, n_viol, dtime])
res_clipping = pd.DataFrame(holder, columns=['lb', 'ub', 'n_viol', 'dtime'])
res_clipping.sort_values('dtime').sort_values('dtime').reset_index(drop=True)

# (2) DOES RUNTIME SCALE WITH SIZE???
tnum = 25
vec_size = [50, 250, 1000, 2500, 5000, 10000]
holder = []
for n in vec_size:
    print(f'n = {n}')
    mu1, mu2, tau21, tau22, a, b = _generate_random_sntn_params(n, seed)
    dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)
    alphas = uniform.rvs(size=n, random_state=seed)
    targets = dist_sntn.Z*alphas
    nt = timeit("_rootfinder_newton(ub=dist_sntn.beta, lb=dist_sntn.alpha, rho=dist_sntn.rho, target_p=targets)", globals=globals(), number=tnum)
    holder.append([nt, n])
res_nvec = pd.DataFrame(holder, columns = ['time', 'size'])
res_nvec.assign(rate=lambda x: (x['size']*tnum / x['time']).astype(int))


#######################################
# --- (5B) FAILURE CASE: USE INIT --- #

# Quantiles for everything!
p_seq = np.arange(0.01, 1, 0.01)

mu1=-0.6117564136500754
tau21=1.6810944055128347
mu2=-0.7612069008951028
tau22=2.791073594843329
a=-1.5449347600694359
b=-0.5311783287768251
c1=0.053362545117080384
c2=0.9466374548829196

# This shoudl all work
dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b, c1=c1, c2=c2)
beta, alpha, rho, theta1, sigma1, Zphi = dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho, dist_sntn.theta1, dist_sntn.sigma1, dist_sntn.Z
quant_loop = np.squeeze(dist_sntn.ppf(p_seq, method='loop'))
cdf_loop = dist_sntn.cdf(quant_loop)
np.testing.assert_allclose(np.squeeze(cdf_loop), p_seq)
cdf_diff = bvn_cdf_diff(x1=(quant_loop-theta1)/sigma1, x2a=beta, x2b=alpha, rho=rho) / Zphi
np.testing.assert_allclose(np.squeeze(cdf_diff), p_seq)
quant_fast = dist_sntn.ppf(p_seq, method='fast')
cdf_fast = dist_sntn.cdf(quant_fast)
np.testing.assert_allclose(np.squeeze(cdf_fast), p_seq)
# But if we turn off smart initialization, it should break
case.assertRaises(RuntimeError, dist_sntn.ppf, p_seq, method='fast', use_approx_init=False)


####################################
# --- (5A) FAILURE CASE: Z ≈ 0 --- #

# Example SNTN parameters
mu1 = -0.2757737286563052
mu2 = -2.6295222258047604
tau21 = 0.10219183113096487
tau22 = 1.414634163492961
a = 1.3885646465047028
b = 3.582846969757054
alpha = 0.7314274410841171

# Work with vinalla approach
dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)
quant_loop = np.atleast_1d(np.squeeze(dist_sntn.ppf(p=alpha, method='loop')))
print(f'Using the loop method, CDF={dist_sntn.cdf(quant_loop)[0]:.4f} is close to alpha={alpha:.4f}')
print(f'But Z={dist_sntn.Z[0]:.4f} is small as beta={dist_sntn.beta[0]:.1f} & alpha={dist_sntn.alpha[0]:.1f}')

try:
    attempt1 = _rootfinder_newton(ub=dist_sntn.beta, lb=dist_sntn.alpha, rho=dist_sntn.rho,
                                    target_p=dist_sntn.Z*alpha, use_hess=True, use_gradclip=False)
except:
    print('Failures with Hessian & w/o gradient clipping')
try:
    attempt2 = _rootfinder_newton(ub=dist_sntn.beta, lb=dist_sntn.alpha, rho=dist_sntn.rho,
                                    target_p=dist_sntn.Z*alpha, use_hess=False, use_gradclip=False)
except:
    print('Failures w/o Hessian & w/o gradient clipping')