"""
Shows how to use the nts class to answer the question posed in the 1964 issue of Technometrics: https://www.jstor.org/stable/1266101?seq=1
"""

# External
import numpy as np
from scipy.stats import norm
# Internal
from sntn.dists import nts

# Parameters of the distribution
mu1, tau21 = 100, 6**2
mu2, tau22 = 50, 3**2
b = np.inf
w = 138
a_seq = np.arange(40, 66)
# Compare conditional to unconditional
dist_uncond = norm(loc=mu1+mu2, scale=np.sqrt(tau21 + tau22))
pval_uncond = dist_uncond.cdf(w)
mean_uncond = dist_uncond.mean()
dist_cond = nts(mu1, tau21, mu2, tau22, a_seq, b)
pval_cond = dist_cond.cdf(w)
mean_cond = dist_cond.mean()

