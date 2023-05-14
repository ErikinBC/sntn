"""
Show that NTS works with a data-carved lasso

python3 -m examples.4_lasso
"""

# External
import os
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm, binom
# Internal
from parameters import dir_figures
from sntn.posi import lasso
from sntn.utilities.linear import dgp_sparse_yX
from sntn.utilities.utils import get_CI, pn_labeller


# Set up simulation parameters
# Dimension of data
n, p = 100, 150
s, k = 5, 10
b0 = +1  # Intercept
alpha = 0.1
lam_max_frac = 0.725
# Number of simulations per snr
nsim = 100
n_snr = 7
frac_split_seq = [0.15, 0.20, 0.25]
n_perm = n_snr*nsim
snr_seq_log10 = np.linspace(-1, +1, n_snr)
snr_seq = np.exp(snr_seq_log10 * np.log(10))

# Suppress selection warnings
import warnings
warnings.simplefilter("ignore")



##############################
# --- (1) RUN SIMULATION --- #

idx = 0
stime = time()
holder_sim, holder_nsel = [], []
for j, snr in enumerate(snr_seq):
    for i in range(nsim):
        # Draw data
        idx += 1
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=idx, return_params=True)
        # Approximate lambda-max
        # Loop over the difference frac to split
        lam_max_approx = np.max(np.abs(np.dot( ((x - x.mean(0))/x.std(0)).T, y)) / len(x) )
        lam_approx = lam_max_approx * lam_max_frac
        for frac_split in frac_split_seq:
            # (i) Split x into screening and split
            split_lasso = lasso(lam_approx, y, x, frac_split=frac_split, seed=idx)
            dat_nsel = pd.DataFrame({'sim':i,'snr':snr, 'p':split_lasso.n_cidx_screened,'frac':frac_split},index=[idx])
            holder_nsel.append(dat_nsel)

            # (ii) Calculate the ground truth parameters on "selected" model
            idx_noise = ~(np.array(split_lasso.cidx_screen) < s)
            n_screened = np.sum(~idx_noise)
            sigma2_gt = 1 + (s-n_screened)*beta1[0]**2
            beta_gt = np.where(idx_noise, 0, beta1[0])

            # (iii) Do naive inference (type-I errors shuold be inflated)
            split_lasso.ols_screen.run_inference(alpha, 0, sigma2_gt)
            ols_naive_i = split_lasso.ols_screen.res_inf.assign(cidx=split_lasso.cidx_screen, mdl='naive')
            ols_naive_i = ols_naive_i[['bhat','pval','cidx','mdl']]
            
            # (iv) Run PoSI inference for screened data
            any_split = frac_split > 0
            split_lasso.run_inference(alpha, 0, sigma2_gt, run_carve=any_split, run_split=any_split, run_ci=False)
            tnorm_screen_i = split_lasso.res_screen.assign(cidx=split_marginal.cidx_screen, mdl='screen')
            
            # nts_carve_i, gauss_split_i = pd.DataFrame({}), pd.DataFrame({})
            # if any_split:
            #     gauss_split_i = split_marginal.res_split.drop(columns=['lb','ub']).assign(mdl='split')
            #     nts_carve_i = split_marginal.res_carve.assign(cidx=split_marginal.cidx_screen, mdl='carve')
            # # Combine and save
            # res_screen_i = pd.concat(objs=[ols_naive_i, tnorm_screen_i, gauss_split_i, nts_carve_i])
            # res_screen_i = res_screen_i.assign(snr=snr, sim=i, frac_split=frac_split)
            # holder_sim.append(res_screen_i)

        breakpoint()
        if idx % 25 == 0:
            dtime = time() - stime
            rate = idx / dtime
            seta = (n_perm - idx)/rate
            print(f'Simulation {idx} of {n_perm} (ETA={seta:0.0f} seconds)')
res_nsel = pd.concat(holder_nsel).reset_index(drop=True)
res_nsel = res_nsel.assign(n_split=lambda x: (x['frac']*n).astype(int))
res_nsel = res_nsel.assign(no_sel=lambda x: x['p']==0)
res_nsel = res_nsel.assign(no_inf=lambda x: x['n_split'] < x['p'] + 1)
print(res_nsel.groupby(['snr','frac'])[['no_sel','no_inf']].mean().round(2))
breakpoint()



        #     # (ii) Calculate the ground truth parameters on "selected" model
        #     idx_noise = ~(np.array(split_marginal.cidx_screen) < s)
        #     n_screened = np.sum(~idx_noise)
        #     sigma2_gt = 1 + (s-n_screened)*beta1[0]**2
        #     beta_gt = np.where(idx_noise, 0, beta1[0])
            
        #     # (iii) Do naive inference (type-I errors shuold be inflated)
        #     split_marginal.ols_screen.run_inference(alpha, 0, sigma2_gt)
        #     ols_naive_i = split_marginal.ols_screen.res_inf.assign(cidx=split_marginal.cidx_screen, mdl='naive')
        #     ols_naive_i = ols_naive_i[['bhat','pval','cidx','mdl']]

        #     # (iv) Run PoSI inference for screened data
        #     any_split = frac_split > 0
        #     split_marginal.run_inference(alpha, 0, sigma2_gt, run_carve=any_split, run_split=any_split, run_ci=False)
        #     tnorm_screen_i = split_marginal.res_screen.assign(cidx=split_marginal.cidx_screen, mdl='screen')
            
        #     nts_carve_i, gauss_split_i = pd.DataFrame({}), pd.DataFrame({})
        #     if any_split:
        #         gauss_split_i = split_marginal.res_split.drop(columns=['lb','ub']).assign(mdl='split')
        #         nts_carve_i = split_marginal.res_carve.assign(cidx=split_marginal.cidx_screen, mdl='carve')
        #     # Combine and save
        #     res_screen_i = pd.concat(objs=[ols_naive_i, tnorm_screen_i, gauss_split_i, nts_carve_i])
        #     res_screen_i = res_screen_i.assign(snr=snr, sim=i, frac_split=frac_split)
        #     holder_sim.append(res_screen_i)
        

