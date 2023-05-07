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
Root finding problem happens because default of Owen's T does not always yield accurate estimates of BVN
"""
x = 5.1526816029843125
tau21 = 5.31504579
tau22 = 1.33290728
rho = np.sqrt(tau22) / np.sqrt(tau21 + tau22)
a = -3.50215787
b = -2.09831269
fix_mu = True

npoints = 25
mu_seq = np.linspace(8, 10, npoints)
pval = np.zeros([npoints, 2])
for i, mu in enumerate(mu_seq):
    dist_mu_scipy = nts(mu, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='scipy')
    dist_mu_owen = nts(mu, tau21, None, tau22, a, b, fix_mu=True, cdf_approach='owen')
    pval_mu_owen = dist_mu_owen.cdf(x)
    pval_mu_scipy = dist_mu_scipy.cdf(x)
    pval[i] = np.concatenate([pval_mu_scipy, pval_mu_owen])
# Errors accross Owen's
pd.DataFrame(pval, columns=['scipy','owen'],index=mu_seq).rename_axis('mu').diff()>0

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


