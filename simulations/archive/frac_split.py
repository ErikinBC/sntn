"""
Understanding what mechanistically happens as the fraction of data going to the split data...
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

# Parameters
alpha = 0.1
nsim = 500
n, p = 150, 250
s, k = 5, 10
cidx_s = list(range(s))
b0 = +1
snr = 1
beta_j = np.sqrt(snr / s)
beta_j2 = snr / s
frac_split_seq = np.round(np.arange(0.0, 0.91, 0.10),2)


##############################
# --- (1) RUN SIMULATION --- #

# NHST
null_beta = 0 
cn_null_beta = ['cidx','pval']

# Loop over different data fractions
stime = time()
holder_bhat, holder_pval, holder_nts = [], [], []
for i in range(nsim):
    # Draw data (will be repeatedly sampled...)
    y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=i, return_params=True)
    assert isclose(beta1[0]**2, beta_j2)
    if (i+1) % 5 == 0:
        dtime = time() - stime
        rate = (i+1) / dtime
        seta = (nsim-i-1)/rate
        print(f'Iteration {i+1} of {nsim} (ETA={seta:0.0f} seconds)')
    break
    for frac_split in frac_split_seq:
        # (i) Generate data
        posi = marginal_screen(k, y, x, frac_split=frac_split, seed=i)
        n_screen = posi.x_screen.shape[0]
        n_split = posi.x_split.shape[0]
        if frac_split == 0.1:
            break
        # print(f'frac={frac_split}: n_screen={n_screen}, n_split={n_split}')
        
        # (ii) Determine ground truth
        cidx = posi.cidx_screen
        idx_tp = np.isin(cidx, range(s))
        n_tp = np.sum(idx_tp)
        sigma2_submdl = 1 + (s-n_tp)*beta_j2
        null_act = np.where(idx_tp, beta_j, 0)

        run_carve, run_split = True, True
        if frac_split == 0:
            run_carve, run_split = False, False
        # (iii) Run inference for "null" hypothesis (this is for p-values)
        posi.run_inference(alpha, null_beta, sigma2_submdl, run_ci=False, run_carve=run_carve, run_split=run_split)
        dat_pval_null = pd.DataFrame({'sim':i+1,'frac':frac_split,'null':'zero','cidx':cidx, 'msr':'pval',  'screen':posi.res_screen.pval})
        if run_carve:
            dat_pval_null['split'] = posi.res_split.pval
            dat_pval_null['carve'] = posi.res_carve.pval
            # Save NTS parameters
            dat_nts_params = pd.DataFrame({'sim':i+1,'frac':frac_split,'cidx':cidx,'tau21':posi.dist_carve.tau21, 'tau22':posi.dist_carve.tau22, 'a':posi.dist_carve.a, 'b':posi.dist_carve.b})
            holder_nts.append(dat_nts_params)

        # (iv) Re-run inference with "correct" null hypothesis
        posi.run_inference(alpha, null_act, sigma2_submdl, run_ci=False, run_carve=run_carve, run_split=run_split)
        dat_pval_act = pd.DataFrame({'sim':i+1,'frac':frac_split,'null':'act', 'cidx':cidx, 'msr':'pval', 'screen':posi.res_screen.pval})
        if run_carve:
            dat_pval_act['split'] = posi.res_split.pval
            dat_pval_act['carve'] = posi.res_carve.pval
        dat_pval_act = dat_pval_act[dat_pval_act['cidx'].isin(cidx_s)]
        
        # Merge and append
        dat_pval = pd.concat(objs=[dat_pval_act, dat_pval_null])
        holder_pval.append(dat_pval)

        # (v) Ge the coefficients
        dat_bhat = pd.DataFrame({'sim':i+1, 'frac':frac_split, 'cidx':cidx, 'screen':posi.res_screen.bhat})
        if run_carve:
            dat_bhat['split'] = posi.res_split.bhat
            dat_bhat['carve'] = posi.res_carve.bhat
            dat_bhat['theory'] = posi.dist_carve.mean()
        holder_bhat.append(dat_bhat)
# Merge
res_bhat = pd.concat(holder_bhat).reset_index(drop=True)
res_bhat['noise'] = ~res_bhat['cidx'].isin(cidx_s)
res_bhat.to_csv('examples/res_bhat.csv',index=False)
res_pval = pd.concat(holder_pval).reset_index(drop=True)
res_pval['noise'] = ~res_pval['cidx'].isin(cidx_s)
res_pval.to_csv('examples/res_pval.csv',index=False)
res_nts = pd.concat(holder_nts).reset_index(drop=True)
res_nts['noise'] = ~res_nts['cidx'].isin(cidx_s)
cn_on = ['sim','frac','cidx']
res_nts = res_nts.merge(res_bhat[cn_on+['split','screen']], 'left', cn_on)
res_nts.to_csv('examples/res_params.csv',index=False)

# # Load for debug
# res_bhat = pd.read_csv('examples/res_bhat.csv')
# res_pval = pd.read_csv('examples/res_pval.csv')
# res_nts = pd.read_csv('examples/res_params.csv')


#################################
# --- (2) CHECK CALIBRATION --- #

cn_methods = ['split', 'screen', 'carve']

# (i) beta's should align (CORRECT)
res_bhat[['carve','theory']].mean()
res_bhat.groupby('noise')[['carve','theory']].mean()

# (ii) We should expect uniform p-values (CORRECT)
cn_gg = ['noise']
dat_uniform = res_pval.query('(null=="zero" & noise) | (null=="act" & ~noise)').reset_index(drop=True)
dat_uniform.groupby('noise')[cn_methods].mean()
dat_uniform.groupby('noise')[cn_methods].quantile([0.025, 0.05, 0.1])

# (iii) Trajectory frac_split
dat_power = res_pval.query('null=="zero" & ~noise').drop(columns=['null','noise']).reset_index(drop=True)
# ON AVERAGE, IT APPEARS THAT TOO MUCH DATA FOR CARVING YOU'D BE BETTER OFF JUST RUNNING CLASSICAL OLS WITH SPLITTING???
(dat_power.set_index('frac')[cn_methods] < alpha).groupby('frac').mean()

# Two open questions: (i) why is carve so much worse than screen at 10%? (ii) Why is split-90 better than carve-90?

res_pval.query('sim==1 & cidx==0 & null=="zero"').drop(columns=['sim','cidx','null'])


res_pval.query('sim==100 & cidx==4').drop(columns='sim')
