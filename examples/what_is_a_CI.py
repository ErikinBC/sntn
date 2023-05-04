# External
import numpy as np
import pandas as pd
from scipy.stats import expon, binom
from scipy.optimize import root_scalar
# Internal
from sntn.dists import tnorm, nts

# --- Monoticity --- #

# dF(theta; phi).cdf(x) / dtheta (strictly positive or negative)

# For truncated normal, theta=mu; phi={sigma2, a, b}
sigma2, a, b, x = 3, -1, 2.3, 0
mu_range = np.arange(-10,11,1)
cdf = np.concatenate([tnorm(mu, sigma2, a, b).cdf(x) for mu in mu_range])
mono_tnorm = pd.DataFrame({'dist':'tnorm','mu':mu_range,'cdf':cdf})

# For NTS, theta={mu1, mu2, or mu==mu1==mu2}, phi={tau21, tau22, a, b, c1(?), c2(?)}
mu1_seq = np.arange(-5,5.5,0.5)
mu2_seq = np.arange(-5,5.5,0.5)
tau21, tau22, a, b, x = 2, 3, -1, 2.3, 0
mono_nts = pd.concat([pd.DataFrame({'mu1':mu1, 'mu2':mu2_seq, 'cdf':np.concatenate([nts(mu1,tau21,mu2,tau22,a,b).cdf(x) for mu2 in mu2_seq])})for mu1 in mu1_seq])
assert np.all(np.sign(mono_nts.groupby('mu1')['cdf'].diff().dropna())==-1), 'For a given mu1, expected the probability to decrease as mu2 increased'
# Repeat for fixed mu
mono_nts_fixed = pd.DataFrame({'mu1':mu1_seq, 'mu2':mu2_seq, 'cdf':np.concatenate([nts(mu1,tau21,None,tau22,a,b,fix_mu=True).cdf(x) for mu1 in mu1_seq])})
assert np.all(np.sign(mono_nts_fixed['cdf'].diff().dropna())==-1), 'Expected the probability to decrease as mu1/2 increased'

# --- Exponential --- #
alpha = 0.05
x = 1.5

# Invert quantile formula
rate_high, rate_low = -np.log(1-np.array([1-alpha/2,alpha/2])) / x
# Check
expon(scale=1/rate_high).cdf(x)
expon(scale=1/rate_low).cdf(x)
# matches interval approach
(rate_low, rate_high)
expon(scale=1/x).interval(0.95)
# But could also use quantile function
expon(scale=1/x).ppf([alpha/2,1-alpha/2])

# --- Binomial --- #
n = 100
p = 0.75
x = n*p
bdist = binom(n=n, p=p)
q_lb, q_ub = bdist.ppf([alpha/2,1-alpha/2]) / n
# Notice this is not exactly the CI we want!
binom(n=n, p=[q_ub, q_lb]).cdf(x)
# Likewise!
binom(n=n, p=np.array(binom.interval(1-alpha,n=n,p=p))/n).cdf(x)
# Whereas what we want to find is...
q_ub_root = root_scalar(lambda pp,nn,xx: binom(n=nn,p=pp).cdf(xx)-alpha/2,args=(n,x),x0=q_lb,x1=q_ub,method='secant').root
q_lb_root = root_scalar(lambda pp,nn,xx: binom(n=nn,p=pp).cdf(xx)-1+alpha/2,args=(n,x),x0=q_lb,x1=q_ub,method='secant').root
print(f'Actual lower-bound: {q_lb_root:.3f}, quantile-approx: {q_lb:.3f}')
print(f'Actual upper-bound: {q_ub_root:.3f}, quantile-approx: {q_ub:.3f}')


binom(n=n, p=0.85).cdf(x)


