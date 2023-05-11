"""
Show that marginal screening works, and what the power is

python3 -m examples.coverage_power
"""


# External
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm, binom
from statsmodels.stats.proportion import proportion_confint
# Internal
from parameters import dir_figures
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX


##############################
# --- (1) RUN SIMULATION --- #

# Dimension of data
n, p = 50, 100
s, k = 5, 10
b0 = +1  # Intercept
alpha = 0.1
# Number of simulations per snr
nsim = 100
n_snr = 7
n_perm = n_snr*nsim
snr_seq_log10 = np.linspace(-1, +1, n_snr)
snr_seq = np.exp(snr_seq_log10 * np.log(10))
df_snr = pd.DataFrame({'noise':False, 'snr':snr_seq, 'beta':np.sqrt(snr_seq / s)})
df_snr = pd.concat(objs=[df_snr.assign(noise=True,beta=0),df_snr])

stime = time()
holder_sim = []
idx = 0
for j, snr in enumerate(snr_seq):
    for i in range(nsim):
        # Draw data
        idx += 1
        if idx % 25 == 0:
            dtime = time() - stime
            rate = idx / dtime
            seta = (n_perm - idx)/rate
            print(f'Simulation {idx} of {n_perm} (ETA={seta:0.0f} seconds)')
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=idx, return_params=True)

        # (i) Marginal screen on all data (no holdout)
        screen_only = marginal_screen(k, y, x, frac_split=0.0)
        idx_noise = ~(np.array(screen_only.cidx_screen) < s)
        n_screened = np.sum(~idx_noise)
        sigma2_gt = 1 + (s-n_screened)*beta1[0]**2
        beta_gt = np.where(idx_noise, 0, beta1[0])
        screen_only.ols_screen.run_inference(alpha, 0, sigma2_gt)
        ols_screen_i = screen_only.ols_screen.res_inf.assign(cidx=screen_only.cidx_screen, mdl='ols') # ,beta=beta_gt
        # Get the exact distribution
        screen_only.get_A()
        screen_only.run_inference(alpha, 0, sigma2_gt)
        tnorm_screen_i = screen_only.res_screen.assign(cidx=screen_only.cidx_screen, mdl='posi') # ,beta=beta_gt
        res_screen_i = pd.concat(objs=[ols_screen_i, tnorm_screen_i]).assign(snr=snr,sim=i,frac_split=0)

        # Merge and store
        holder_sim.append(res_screen_i)
res_sim = pd.concat(holder_sim).reset_index(drop=True)
res_sim['noise'] = ~(res_sim['cidx'] < s)
res_sim = res_sim.merge(df_snr,'left')
res_sim = res_sim.assign(cover=lambda x: (x['lb'] <= x['beta']) & (x['ub'] >= x['beta']))
res_sim = res_sim.assign(reject=lambda x: x['pval'] < alpha)


############################
# --- (2) PLOT RESULTS --- #

# (i) Calculate coverage
cn_gg = ['mdl','snr','frac_split']
res_sim.groupby(['noise'] + cn_gg)['cover'].agg({'mean'}).round(3)

# (ii) Calculate power
res_sim.groupby(['noise'] + cn_gg)['reject'].agg({'mean'})

# (iii) Calculate selection prob
res_sim.groupby(cn_gg)['noise'].agg({'sum','count'}).assign(n=lambda x: x['count']-x['sum'],tot=lambda x: s*nsim).drop(columns=['count','sum']).assign(pct=lambda x: x['n']/x['tot'])


import plotnine as pn
