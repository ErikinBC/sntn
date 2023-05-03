"""
Shows how to use the nts class to answer the question posed in the 1964 issue of Technometrics: https://www.jstor.org/stable/1266101?seq=1

python3 -m examples.technometrics
"""

# External
import numpy as np
from scipy.stats import norm
# Internal
from sntn.dists import nts
from sntn._cdf_bvn._utils import cdf_to_orthant

# Parameters of the distribution
mu1, tau21 = 100, 6**2
mu2, tau22 = 50, 3**2
b = np.inf
w = 138
a_seq = np.arange(40, 66)
# a_seq = np.concatenate((np.array([-np.inf]),np.arange(44, 66)))

norm(loc=mu1+mu2, scale=np.sqrt(tau21 + tau22)).cdf(w)

# Get the range of means
mu_of_a = np.array([nts(mu1, tau21, mu2, tau22, a, b).mean()[0] for a in a_seq])
for a in a_seq:
    dist_a = nts(mu1, tau21, mu2, tau22, a, b, cdf_approach='owen')
    pval = dist_a.cdf(w)[0]
    print(f'pval={pval:.5f}, a={a:0.0f}')
    