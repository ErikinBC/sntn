"""
Approximations and dealing with tails

python3 -m examples.orthant_tails
"""

# Internal
import numpy as np
import pandas as pd
from scipy.stats import norm
# External
from sntn.dists import nts, bvn
from sntn.utilities.grad import _log_gauss_approx

"""
Another issue
"""

mu = 0
tau21 = 4
tau22 = 0.04040404040404041
a = 1
b = np.inf
c1, c2 = 0.01, 0.99
dist = nts(mu, tau21, mu, tau22, a, b, c1, c2, cdf_approach='scipy')
# dist.mean()
# dist.rvs(1000000).mean()
x_seq = np.linspace(1,2,30)
dist.cdf(x_seq)


"""
Another issue
"""

# mu = 
x = -5.42473206
tau21 = 11.11613935
tau22 = 0.03412925
a = -0.34085213
b = -0.28341952
m_seq = np.linspace(-33, 33.1, 21)
cdf = nts(m_seq, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='scipy').cdf(x).flatten()
print(pd.DataFrame({'cdf':cdf,'mu':m_seq}))

"""
Another issue
"""

x = 0.09308509
mu = -1.0392994
tau21 = 0.03343167
tau22 = 0.03407728
a = 0.22111192
b = 0.37452124
c1, c2 = 0.5, 0.5
nts(mu, tau21, None, tau22, a, b, c1, c2, fix_mu=True, cdf_approach='scipy').cdf(x)
nts(mu, tau21, None, tau22, a, b, c1, c2, fix_mu=True, cdf_approach='owen').cdf(x)


"""
Coverage issue??
"""
alpha = 0.10
mu = -1.0981486325572187
a = -2.7137160197677566
b = -0.19826792238746457
tau21 = 43.10846981486246
tau22 = 2.268561623113654
x = 3.535477
lb, ub = -26.16196, -5.192813

dist_gt = nts(mu, tau21, None, tau22, a=a, b=b, fix_mu=True, cdf_approach='scipy')

dist_gt.ppf(alpha/2)
dist_gt.ppf(1-alpha/2)
ci_emp = dist_gt.conf_int(x, alpha=alpha, cdf_approach='owen').flatten()
nts(ci_emp[0], tau21, None, tau22, a=a, b=b, fix_mu=True, cdf_approach='scipy').cdf(x)
nts(ci_emp[1], tau21, None, tau22, a=a, b=b, fix_mu=True, cdf_approach='scipy').cdf(x)



"""
Unstable CDF?
"""

x = 2.91940025
tau21 = 1.12743604
tau22 = 1.84161871
a = 0.50008805
b = 2.29270529
mu_seq = np.arange(-17, 18)
dist_nts = nts(mu_seq, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='scipy')
dist_nts.cdf(x)


"""
Root finding problem happens because default of Owen's T does not always yield accurate estimates of BVN
"""
x = 0.51107317
# x = 5.1526816029843125
tau21 = 5.31504579
tau22 = 1.33290728
rho = np.sqrt(tau22) / np.sqrt(tau21 + tau22)
a = -3.50215787
b = -2.09831269
fix_mu = True

npoints = 25
mu_seq = np.linspace(-5, 5, npoints)
pval = np.zeros([npoints, 2])
for i, mu in enumerate(mu_seq):
    dist_mu_scipy = nts(mu, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='scipy')
    dist_mu_owen = nts(mu, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='owen')
    pval_mu_owen = dist_mu_owen.cdf(x)
    pval_mu_scipy = dist_mu_scipy.cdf(x)
    pval[i] = np.concatenate([pval_mu_scipy, pval_mu_owen])
# Errors accross Owen's
df = pd.DataFrame(pval, columns=['scipy','owen'],index=mu_seq)
df
df.rename_axis('mu').diff()>0

m1 = (x - 2*mu_seq[13])/np.sqrt(tau21 + tau22)
m2 = (x - 2*mu_seq[14])/np.sqrt(tau21 + tau22)
alpha1 = (a - mu_seq[13]) /np.sqrt(tau22)
alpha2 = (a - mu_seq[14]) /np.sqrt(tau22)
beta1 = (b - mu_seq[13]) /np.sqrt(tau22)
beta2 = (b - mu_seq[14]) /np.sqrt(tau22)
Z1 = norm.cdf(beta1) - norm.cdf(alpha1)
Z2 = norm.cdf(beta2) - norm.cdf(alpha2)
pd.DataFrame({'m':[m1,m2],'alpha':[alpha1,alpha2],'beta':[beta1,beta2],'Z':[Z1,Z2]})
dist_bvn = bvn(0,1,0,1,rho, cdf_approach='scipy')
(dist_bvn.cdf(x1=m1, x2=beta1)-dist_bvn.cdf(x1=m1, x2=alpha1))/Z1
(dist_bvn.cdf(x1=m2, x2=beta2)-dist_bvn.cdf(x1=m2, x2=alpha2))/Z2
dist_bvn = bvn(0,1,0,1,rho, cdf_approach='owen')
(dist_bvn.cdf(x1=m1, x2=beta1)-dist_bvn.cdf(x1=m1, x2=alpha1))/Z1
(dist_bvn.cdf(x1=m2, x2=beta2)-dist_bvn.cdf(x1=m2, x2=alpha2))/Z2





"""
Orthant prob transform matters
"""

rho = 0.4477710354540847
m1 = -5.044314647627817
alpha = -9.962752794866882
beta = -8.746792913124041
# m1 = 0.5
# alpha = -2
# beta = -1
dist = bvn(0,1,0,1,rho, cdf_approach='owen')

# P(X > h, X > k) = 1 - (Phi(h) + Phi(k)) + MVN(X <= h, X<= k)
orth1 = 1 - (norm.cdf(m1) + norm.cdf(alpha)) + dist.cdf(x1=m1, x2=alpha)
orth2 = 1 - (norm.cdf(m1) + norm.cdf(beta)) + dist.cdf(x1=m1, x2=beta)
Z = norm.cdf(beta) - norm.cdf(alpha)
1 - (orth1 - orth2) / Z
(dist.cdf(x1=m1, x2=beta) - dist.cdf(x1=m1, x2=alpha)) / Z


