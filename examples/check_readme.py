"""
Makes sure that the python code in the README works as 
expected
"""

import numpy as np
from sntn.dists import nts
from sntn.posi import lasso

## (i) Filtering regime
mu1, tau21 = 100, 6**2
mu2, tau22 = 50, 3**2
a, b = 44, np.inf
w = 138
dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
cdf_1964 = dist_1964.cdf(w)[0]
print(f'Probability that Z<138={cdf_1964*100:0.3f}%')


### (ii) Data carving
seed = 2
alpha = 0.05
bhat_null = 0
n, p, s = 100, 150, 12
beta0 = np.zeros(p)
b = 0.35
beta0[:s] = b
null_beta = 0
np.random.seed(seed)
x = np.random.randn(n,p)
g = x.dot(beta0)
u = np.random.randn(n)
y = g + u
lammax = np.max(np.abs(x.T.dot(y)/n))
lam = lammax * 0.5
# Lasso only
inf_posi = lasso(lam, y, x, frac_split=0)
inf_posi.run_inference(alpha, null_beta, sigma2=1.0, run_carve=False, run_split=False, run_ci=False)
inf_posi = inf_posi.res_screen.query('pval < @alpha')
idx_tp_lasso = inf_posi['cidx'] < s
tp_lasso, fp_lasso = np.sum(idx_tp_lasso), np.sum(~idx_tp_lasso)
# 50/50 split
inf_split = lasso(lam, y, x, frac_split=0.5, seed=seed)
inf_split.run_inference(alpha, null_beta, sigma2=1.0, run_carve=False, run_ci=False)
inf_split = inf_split.res_screen.query('pval < @alpha')
idx_tp_split = inf_split['cidx'] < s
tp_split, fp_split = np.sum(idx_tp_split), np.sum(~idx_tp_split)
print(f'Lasso (TP={tp_lasso}, FP={fp_lasso})\nSplit (TP={tp_split}, FP={fp_split})')
# 90/10 carve
inf_carve = lasso(lam, y, x, frac_split=0.1, seed=seed)
inf_carve.run_inference(alpha, null_beta, sigma2=1.0, run_ci=False)
inf_carve = inf_carve.res_carve.query('pval < @alpha')
idx_tp_carve = inf_carve['cidx'] < s
tp_carve, fp_carve = np.sum(idx_tp_carve), np.sum(~idx_tp_carve)
print(f'Lasso (TP={tp_lasso}, FP={fp_lasso})\nSplit (TP={tp_split}, FP={fp_split})\nCarve (TP={tp_carve}, FP={fp_carve})')
