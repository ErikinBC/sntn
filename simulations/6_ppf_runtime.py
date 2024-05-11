"""
Do experiments on ppf solving runtime

python3 -m simulations.6_ppf_runtime
"""

# Load modules
import numpy as np
import pandas as pd
from sntn import dists
from time import time
from timeit import timeit
from scipy.stats import chi2, norm
# Define parameters
n = 96
m = 158
sigma2 = 2.7
k = 0.5
dof = n + m - 1
Delta = k * np.sqrt(sigma2 / n)
seed = 1234


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

nvec = 101
m_seq = np.linspace(-0.5, 0.5, nvec)
delta_seq = np.linspace(1, 2, nvec)
omega_seq = delta_seq - 2
dist_BVN.cdf(x1=m_seq, x2=delta_seq) - dist_BVN.cdf(x1=m_seq, x2=omega_seq)

bvn_cdf_diff(x1=m_seq, x2a=delta_seq, x2b=omega_seq, rho=rho, n_points=1001)

from timeit import timeit
num = 25
# Note that about 5000 iterations matches cdf...
timeit("(dist_BVN.cdf(x1=m, x2=delta) - dist_BVN.cdf(x1=m, x2=omega))[0]", number=num, globals=globals())
timeit("bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho, n_points=1001)", number=num, globals=globals())





import sys
sys.exit('stop here')



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


