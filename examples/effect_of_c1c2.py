"""
Show why c1c2 is important

python3 -m examples.effect_of_c1c2

"""


# External
import os
import numpy as np
import pandas as pd
from time import time
from math import isclose
from scipy.stats import norm, binom
# Internal
from parameters import seed
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX


############################
# --- (1) SAMPLE MEANS --- #

k = 10
mu = np.repeat(1, k)
mu = np.repeat(2, k)
sigma = np.sqrt(sigma2)
n = 50
frac_split = 0.1

# Set up oracle distribution
dist_oracle = norm(mu, sigma2)

# Generate actual data
# x = 


# Calculate sample mean..
# xbar = 



##################################
# --- (2) MARGINAL SCREENING --- #

# Parameters
seed = 0
alpha = 0.1
n, p = 150, 250
s, k = 5, 10
cidx_s = list(range(s))
b0 = +1
snr = 1
frac_split = 0.1

# (i) Draw data
y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=seed, return_params=True)
posi = marginal_screen(k, y, x, frac_split=frac_split, seed=seed)

# (ii) Determine ground truth
cidx = posi.cidx_screen
idx_tp = np.isin(cidx, range(s))
n_tp = np.sum(idx_tp)
sigma2_submdl = 1 + (s-n_tp)*beta_j2
null_act = np.where(idx_tp, beta_j, 0)
null_beta = np.zeros(null_act.shape)

# (iii) Calculate unweighted inference
posi.run_inference(alpha, null_beta, sigma2_submdl, c1=1, c2=1, run_ci=False)
dist_nts_unw = posi.dist_carve
dist_tnorm_unw = posi.dist_screen
pval_raw = pd.DataFrame({'cidx':cidx, 'screen':posi.res_screen.pval, 'split':posi.res_split.pval, 'carve':posi.res_carve.pval})
pval_raw = pval_raw.sort_values('cidx').query('cidx < @s').reset_index(drop=True)

# (iv) Calculate weighted inference
posi.run_inference(alpha, null_beta, sigma2_submdl, c1=frac_split, c2=1-frac_split, run_ci=False)
dist_nts_w = posi.dist_carve
dist_tnorm_w = posi.dist_screen
pval_w = pd.DataFrame({'cidx':cidx, 'screen':posi.res_screen.pval, 'split':posi.res_split.pval, 'carve':posi.res_carve.pval})
pval_w = pval_raw.sort_values('cidx').query('cidx < @s').reset_index(drop=True)

# tnorm should align, nts should not...
x = 0
assert np.all(dist_nts_w.cdf(x) != dist_nts_unw.cdf(x))
assert np.all(dist_tnorm_unw.cdf(x) == dist_tnorm_w.cdf(x))


