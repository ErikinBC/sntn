"""
Do experiments on ppf solving runtime

python3 -m simulations.6_ppf_runtime
"""

# External modules
import numpy as np
import pandas as pd
from time import time
from timeit import timeit
from scipy.stats import chi2, norm, uniform, expon
# Internal modules
from sntn import dists
from sntn._quad import bvn_cdf_diff

seed = 1234
tnum = 100


##########################
# --- (1) M-QUANTILE --- #

# note that in the SNTN CDF, m1(z) = (z - theta1)/sigma1
#       so z(m1) = sigma1*m1 + theta1

# Some parameters
nsim = 1
alpha = 0.13

# Generate data
mu1 = norm.rvs(size=nsim, random_state=seed)
mu2 = norm.rvs(size=nsim, random_state=seed+1)
tau21 = expon(scale=1).rvs(size=nsim, random_state=seed)
tau22 = expon(scale=1).rvs(size=nsim, random_state=seed+1)
a = uniform(loc=-3, scale=1).rvs(size=nsim, random_state=seed)
b = a + 4

# Check that default PPF works as expected
dist_sntn = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=a, b=b)
quant_p = np.atleast_1d(np.squeeze(dist_sntn.ppf(p=alpha)))
p_quant = np.atleast_1d(np.squeeze(dist_sntn.cdf(quant_p)))
print(f'Calculated quantile={quant_p[0]:.3f} for p-value ({alpha:.2f}), and cdf={p_quant[0]:.3f}')

# Check it can be recovered with bvn_cdf_diff
m_p = (quant_p - dist_sntn.theta1) / dist_sntn.sigma1
target_p = dist_sntn.Z * alpha
# np.testing.assert_allclose(bvn_cdf_diff(m_p, dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho) / dist_sntn.Z , alpha)
np.testing.assert_allclose(bvn_cdf_diff(m_p, dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho) , target_p)

# Quick and dirty root finding
from scipy.optimize import root_scalar

rootfun = lambda z, ub, lb, rho, p: bvn_cdf_diff(z, ub, lb, rho) - p

m_p_root = root_scalar(f=rootfun, args=(dist_sntn.beta, dist_sntn.alpha, dist_sntn.rho, target_p), method='brentq', bracket=(-6, 6)).root
np.testing.assert_allclose(m_p, m_p_root)
quant_root = m_p_root*dist_sntn.sigma1 + dist_sntn.theta1
np.testing.assert_allclose(quant_root, quant_p)


###########################
# --- (1) FASTEST CDF --- #

from sntn._bvn import _bvn as bvn
from scipy.stats import multivariate_normal as mvn

# Check the scalar
rho = 0.62
delta = 1.0
omega = -2.0
m = -0.2

bvn_owen = bvn(0, 1, 0, 1, rho, cdf_approach='owen')
q1 = bvn_owen.cdf(x1=m, x2=delta) - bvn_owen.cdf(x1=m, x2=omega)
q2 = bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho)
print(np.abs(q1 - q2).max())

# Check speed on vector
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

# Check accuracy on difference
print(np.abs(bvn_cdf_diff(x1=x1, x2a=delta, x2b=omega, rho=rho) - (bvn_owen.cdf(x1=x1, x2=delta) - bvn_owen.cdf(x1=x1, x2=omega))).max())

tnum = 1000
tdiff = timeit("bvn_cdf_diff(x1=x1, x2a=delta, x2b=omega, rho=rho)", globals=globals(), number=tnum)
towen = timeit("bvn_owen.cdf(x1=x1, x2=delta) - bvn_owen.cdf(x1=x1, x2=omega)", globals=globals(), number=tnum)
print(f'Run-time for vector of length {n}, {tnum} times:\nbvn_cdf_diff={tdiff:0.2f}, bvn_owen={towen:0.2f}')

# timeit("bvn_owen.cdf(x1=x1, x2=x2)", globals=globals(), number=tnum)
# timeit("bvn_scipy.cdf(x1=x1, x2=x2)", globals=globals(), number=tnum)
# timeit("bvn_mvn.cdf(x)", globals=globals(), number=tnum)

import sys
sys.exit('stop here')

################################
# --- (X) QUANTILE FINDING --- #

# Define parameters
n = 96
m = 158
sigma2 = 2.7
k = 0.5
dof = n + m - 1
Delta = k * np.sqrt(sigma2 / n)

from sntn._bvn import _bvn as bvn
from sntn._quad import dbvn_cdf_diff
from scipy.optimize import root_scalar

nparams = 13

mu1 = norm.rvs(size=nparams, random_state=seed)
mu2 = norm.rvs(size=nparams, random_state=seed+1)
sigma21 = chi2(df=1).rvs(size=nparams, random_state=seed)
sigma22 = chi2(df=1).rvs(size=nparams, random_state=seed+1)
rho = uniform.rvs(size=nparams, random_state=seed)

dist_nts = bvn(mu1, sigma21, mu2, sigma22, rho)
dist_nts.rho


############################
# --- (0) CLOSED FORM? --- #

from sntn._bvn import _bvn as bvn
from scipy.integrate import quad
from sntn._quad import bvn_cdf_diff, dbvn_cdf_diff

def integrand_X2(x2, x1, rho) -> float | np.ndarray:
    """Note that x1 here is fixed"""
    return norm.cdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)

def dx1_integrand_X2(x2, x1, rho) -> float | np.ndarray:
    """Note that x1 here is fixed"""
    return (1 / np.sqrt(1-rho**2)) * norm.pdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)

# We know rho, delta, and omega
rho = 0.62
delta = 1.0
omega = -2.0
dist_BVN = bvn(0, 1, 0, 1, rho)
m = -0.2
target = 0.15
# Will be the same...
dcdf1 = (dist_BVN.cdf(x1=m, x2=delta) - dist_BVN.cdf(x1=m, x2=omega))[0]
dcdf2 = quad(func=integrand_X2, a=omega, b=delta, args=(m, rho, ))[0]
dcdf3 = bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho, n_points=501)
print(dcdf1); print(dcdf2); print(dcdf3)

mseq = np.linspace(-0.5, 5, 101)
eps = 1e-5
grad1 = dbvn_cdf_diff(x1=mseq, x2a=delta, x2b=omega, rho=rho)
grad2 = ((dist_BVN.cdf(x1=mseq+eps, x2=delta) - dist_BVN.cdf(x1=mseq+eps, x2=omega)) - (dist_BVN.cdf(x1=mseq-eps, x2=delta) - dist_BVN.cdf(x1=mseq-eps, x2=omega))) / (2*eps)
print(pd.DataFrame({'m':mseq, 'dcdf':dist_BVN.cdf(x1=mseq, x2=delta) - dist_BVN.cdf(x1=mseq, x2=omega), 'grad1':grad1, 'grad2':grad2}).assign(slope=lambda x: x['dcdf'].diff() / (mseq[1] - mseq[0])).round(4).dropna().head(20))


#####################################
# !!! VECTORIZE DIFFERENCE IN CDF !!!

# vectorized
nvec = 107
m_seq = np.linspace(-0.5, 0.5, nvec)
delta_seq = np.linspace(1, 2, nvec)
omega_seq = delta_seq - uniform(loc=1, scale=2).rvs(nvec,random_state=seed)
vec_scipy = dist_BVN.cdf(x1=m_seq, x2=delta_seq) - dist_BVN.cdf(x1=m_seq, x2=omega_seq)
vec_cust = bvn_cdf_diff(x1=m_seq, x2a=delta_seq, x2b=omega_seq, rho=rho, n_points=501)
pd.DataFrame({'scipy':vec_scipy, 'cust':vec_cust}).round(4)

timeit("dist_BVN.cdf(x1=m_seq, x2=delta_seq) - dist_BVN.cdf(x1=m_seq, x2=omega_seq)", number=num, globals=globals())
timeit("bvn_cdf_diff(x1=m_seq, x2a=delta_seq, x2b=omega_seq, rho=rho, n_points=501)", number=num, globals=globals())

# univariate Note that about 5000 iterations matches cdf...
timeit("(dist_BVN.cdf(x1=m, x2=delta) - dist_BVN.cdf(x1=m, x2=omega))[0]", number=num, globals=globals())
timeit("bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho, n_points=1001)", number=num, globals=globals())


############################
# --- (1) ROOT VS LOOP --- #

# QUESTION: How the size of the number of quantiles to find impact the runtime?
quantile = 0.13
niter = 5
holder = []
nsims = np.arange(50, 250+1, 50)
for nsim in nsims:
    print(nsim)
    # Generate some random parameters
    sigma2_hat = sigma2 * chi2(df=dof).rvs(nsim, random_state=seed) / dof
    mu2 = -Delta / np.sqrt(sigma2_hat / m)
    mu1 = np.random.randn(nsim)
    tau21 = np.random.exponential(nsim)
    tau22 = np.random.exponential(nsim)
    dist_fixed = dists.nts(mu1=0, tau21=1, mu2=mu2, tau22=m/n, a=0, b=np.infty)
    dist_rand = dists.nts(mu1=mu1, tau21=tau21, mu2=mu2, tau22=tau22, a=0, b=np.infty)
    time_rand = timeit("dist_rand.ppf(quantile, method='root')", number=niter, globals=globals())
    time_fixed = timeit("dist_fixed.ppf(quantile, method='root')", number=niter, globals=globals())
    time_loop = timeit("dist_fixed.ppf(quantile, method='loop')", number=niter, globals=globals())
    holder.append(pd.DataFrame({'rand':time_rand, 'fixed':time_fixed, 'loop':time_loop},index=[nsim]))
res_runtime = pd.concat(holder).rename_axis('nsolution').\
    melt(ignore_index=False,var_name='method',value_name='time').\
    reset_index().\
    assign(rate=lambda x: (x['nsolution'] * niter) / x['time'])
res_runtime.\
    pivot(index='nsolution', columns='method', values='rate').\
    astype(int)

# ANSWER: 'loop' optional maintains around 130 solutions per second, but root maxes out at a rate of ~1600 solutions per second, WHEN, solving for ~100 roots at a time
nsim = 13000
sigma2_hat = sigma2 * chi2(df=dof).rvs(nsim, random_state=seed) / dof
mu2 = -Delta / np.sqrt(sigma2_hat / m)
dist_fixed = dists.nts(mu1=0, tau21=1, mu2=mu2, tau22=m/n, a=0, b=np.infty)
stime = time()
qhat1 = dist_fixed.ppf(p=quantile, method='loop').mean()
dtime_loop = time() - stime
# stime = time()
# qhat2 = dist_fixed.ppf(p=quantile, method='root').mean()
# dtime_root = time() - stime
print(f'Calculate {nsim} quantiles:\n {dtime_loop:0.0f} seconds w/ loop\n {dtime_loop:0.0f} seconds w/ root')


