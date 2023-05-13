"""
Shows that the BVN method works the same as the scipy.stats.multivariate_normal distribution

python3 -m examples.scipy_comp
"""

# External
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn
# Internal
from sntn.dists import bvn
from sntn._cdf_bvn._utils import cdf_to_orthant

# Parameters to test
rho = 0.447
Sigma = np.array([[1,rho],[rho,1]])
X12 = np.array([[-1.78, 0.0],[+1.78, 0.0],[0.0, -1.78],[0.0, +1.78],
          [+0.5,+0.5],[-0.5,+0.5],[+0.5,-0.5],[-0.5,-0.5]])
holder = []
for i in range(len(X12)):
    x1, x2 = X12[i]
    x = np.array([x1, x2])
    # Baseline
    cdf_scipy = mvn(cov=Sigma).cdf(x)
    orth_scipy = cdf_to_orthant(cdf_scipy, x1, x2)
    # BVN-scipy
    cdf_bvn_scipy = bvn(0, 1, 0, 1, rho, cdf_approach='scipy').cdf(x1=x1, x2=x2)[0]
    orth_bvn_scipy = cdf_to_orthant(cdf_bvn_scipy, x1, x2)
    # BVN-cox
    cdf_bvn_cox = bvn(0, 1, 0, 1, rho, cdf_approach='cox1', nsim=100000).cdf(x1=x1, x2=x2)[0]
    orth_bvn_cox = cdf_to_orthant(cdf_bvn_cox, x1, x2)
    # BVN-Drezner
    cdf_bvn_drez = bvn(0, 1, 0, 1, rho, cdf_approach='drezner1').cdf(x1=x1, x2=x2)[0]
    orth_bvn_drez = cdf_to_orthant(cdf_bvn_drez, x1, x2)
    # BVN-Owen
    cdf_bvn_owen = bvn(0, 1, 0, 1, rho, cdf_approach='owen').cdf(x1=x1, x2=x2)[0]
    orth_bvn_owen = cdf_to_orthant(cdf_bvn_owen, x1, x2)
    # Store
    tmp = pd.DataFrame({'approach':['scipy','bvn-scipy', 'bvn-cox','bvn_drezner','bvn_owen'],
                'cdf':[cdf_scipy, cdf_bvn_scipy, cdf_bvn_cox, cdf_bvn_drez, cdf_bvn_owen],
                'orthant':[orth_scipy, orth_bvn_scipy, orth_bvn_cox, orth_bvn_drez, orth_bvn_owen],
                'x1':x1, 'x2':x2})
    holder.append(tmp)
# Compare
res = pd.concat(holder).reset_index(drop=True)
res = res.pivot(index=['x1','x2'],columns='approach',values='cdf').drop(columns=['scipy']).round(3)
print(res)
