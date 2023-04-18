import numpy as np
from scipy.stats import expon, binom
from scipy.optimize import root_scalar

rate = 5
scale = 1 / rate
dist = expon(scale=scale)
dist.mean()

# Check CI
alpha = 0.05
x = 1.5

# --- Exponential --- #
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


