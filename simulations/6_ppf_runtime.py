"""
Do experiments on ppf solving runtime
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

def integrand_X1(x1, x2, rho) -> float | np.ndarray:
    """Note that x1 here is fixed"""
    return norm.cdf((x2 - rho*x1)/np.sqrt(1-rho**2) ) * norm.pdf(x1)

def _integrand_X12(x1, x2, rho) -> float | np.ndarray:
    """See bvn_cdf_diff"""
    return norm.cdf((x1 - rho*x2)/np.sqrt(1-rho**2) ) * norm.pdf(x2)

def bvn_cdf_diff(x1, x2a, x2b, rho, n_points: int=1001) -> float | np.ndarray:
    """
    Calculates the difference in the CDF between two bivariate normals with a shared x1 and rho value:

    BVN(rho).cdf(x1, x2a) - BVN(rho).cdf(x1, x2b)

    Background
    ==========
    BVN(rho).cdf(x1, x2) = int_{-infty}^{x2} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    Since X1 | X2=z ~ N(rho*z, 1-rho^2)

    So the difference in the integrals is simply:
    int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz
    """
    d_points = 1 / (n_points - 1)
    points = np.linspace(x2b, x2a, num=n_points)
    y = _integrand_X12(x1=x1, x2=points, rho=rho)
    int_f = np.trapz(y, dx=d_points)
    return int_f


def dbvn_cdf_diff(x1, x2a, x2b, rho) -> float | np.ndarray:
    """
    Calculates the derivative of the integral:

    I(x1; x2a, x2b, rho) = int_{x2b}^{x2a} Phi((x1 - rho*z)/sqrt(1-rho^2)) * phi(z) dz

    w.r.t x1:
    
    dI/dx1 = 1/sqrt(1-rho^2) int_{x2b}^{x2a} phi((x1-rho*z)/sqrt(1-rho^2)) phi(z) dz

    This nicely has a closed form solution (see Owen 1980)
    """
    def integral110(x, a, b):
        fb = np.sqrt(1 + b**2)
        res = (1/fb) * norm.pdf(a / fb) * norm.cdf(x*fb + a / fb)
        return res

    frho = np.sqrt(1-rho**2)
    a = x1 / frho
    b = -rho / frho
    val = (integral110(x2a, a, b) - integral110(x2b, a, b)) / frho
    return val


# We know rho, delta, and omega
rho = 1/3
delta = 22.50
omega = 0.25
dist_BVN = bvn(0, 1, 0, 1, rho)
m = 0.2
target = 0.15
# Will be the same...

dcdf1 = (dist_BVN.cdf(x1=m, x2=delta) - dist_BVN.cdf(x1=m, x2=omega))[0]
dcdf2 = quad(func=integrand_X1, a=omega, b=delta, args=(m, rho, ))[0]
dcdf3 = bvn_cdf_diff(x1=m, x2a=delta, x2b=omega, rho=rho, n_points=1001)
dcdf1; dcdf2; dcdf3
points = np.linspace(omega, delta, 10001)
np.trapz(integrand_X1(x1=points, x2=m, rho=rho), x=points)
np.sum(integrand_X1(x1=points, x2=m, rho=rho) * (delta - omega) / 10000)


bvn_cdf_diff(x1=0.11, x2a=delta, x2b=omega, rho=rho, n_points=101) - bvn_cdf_diff(x1=0.10, x2a=delta, x2b=omega, rho=rho, n_points=101)

bvn_cdf_diff(x1=0.10, x2a=delta, x2b=omega, rho=rho) + 0.01*dbvn_cdf_diff(x1=0.10, x2a=delta, x2b=omega, rho=rho)


eps = 1e-10
np.array([(quad(func=integrand_X1, a=omega, b=delta, args=(z+eps, rho, ))[0] - quad(func=integrand_X1, a=omega, b=delta, args=(z-eps, rho, ))[0]) / (2*eps) for z in np.linspace(-0.5, 0.5)])







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


