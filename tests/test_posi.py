"""
Make sure PoSI classes work as expected

python3 -m tests.test_posi
python3 -m pytest tests/test_posi.py -s
"""

# External
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import ElasticNetCV
# Internal
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX, ols


# Data generating process
n, p = 100, 150
b0, s,  = -1, 5
cidx_gt = pd.Series(range(s))
snr = 1
# Screening values
k = 5
# Simulation params
nsim = 750

# If you fix sigma2, but the model is wrong, it doesn't mean the same thing....
holder_sim = []
for i in range(nsim):
    # Draw data
    if (i+1) % 250 == 0:
        print(f'Simulation {i+1} of {nsim}')
    y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, return_params=True, seed=i)
    screener = marginal_screen(k, y, x, seed=i)
    
    # Extract the coefficients
    beta_screen_i = screener.ols_screen.bhat[1:]
    beta_carve_i = screener.ols_carve.bhat[1:]
    
    # Assume we know the oracle variance var(y-f(xbeta))
    n_screened = cidx_gt.isin(screener.cidx_screen).sum()
    sigma2_posi = 1 + (s - n_screened)*beta1[0]**2
    
    # Use the groundtruth variance
    gt_screen_i = np.sqrt(sigma2_posi * screener.ols_screen.igram.diagonal()[1:])
    gt_carve_i = np.sqrt(sigma2_posi * screener.ols_carve.igram.diagonal()[1:])
    
    # Use estimated variance
    screener.estimate_sigma2()
    se_screen_i = screener.se_bhat_screen[1:]
    se_carve_i = screener.se_bhat_carve[1:]
    
    # Save for later
    sim_i = pd.DataFrame({'gt':beta1[:s],'bhat_screen':beta_screen_i, 'bhat_carve':beta_carve_i,'se_screen':se_screen_i, 'se_carve':se_carve_i, 'gt_screen':gt_screen_i, 'gt_carve':gt_carve_i},index=screener.cidx_screen).assign(sim=i)
    holder_sim.append(sim_i)
# Merge data
cn_idx = ['sim','cidx','gt']
res_sim = pd.concat(holder_sim).rename_axis('cidx').reset_index().melt(cn_idx,var_name='tmp')
tmp = res_sim['tmp'].str.split(pat='\\_',n=1,expand=True).rename(columns={0:'param',1:'tt'})
res_sim = pd.concat(objs=[res_sim.drop(columns='tmp'), tmp], axis=1)
# Calculate naive z's
res_sim = res_sim.pivot(index=cn_idx+['tt'],columns='param',values='value')
res_sim = res_sim.assign(z_est=lambda x: x['bhat']/x['se'], z_gt=lambda x: x['bhat']/x['gt'])
res_sim.drop(columns=['se','gt'], inplace=True)
res_sim = res_sim.melt('bhat', ignore_index=False, var_name='sigma2',value_name='z')
res_sim['sigma2'] = res_sim['sigma2'].str.replace('z_','',regex=False)
res_sim = res_sim.assign(pval=lambda x: 2*norm.cdf(-x['z'].abs()))
res_sim = res_sim.reset_index().assign(noise = lambda x:  ~x['cidx'].isin(range(s)))
# Clearly inflated...
cn_gg = ['noise','tt','sigma2']
# Check that roughlt standard normal z, and roughly uniform p
res_sim.groupby(cn_gg).agg({'z':['mean','std'], 'bhat':['mean','std']})
res_sim.groupby(cn_gg)['pval'].quantile(np.arange(0.1,1,0.1)).round(2).reset_index().rename(columns={f'level_{len(cn_gg)}':'quantiles'}).pivot(index=cn_gg,columns='quantiles',values='pval').T