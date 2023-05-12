"""
Show that marginal screening works, and what the power is

python3 -m examples.coverage_power
"""


# External
import os
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm, binom
# Internal
from parameters import dir_figures
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX
from sntn.utilities.utils import get_CI, pn_labeller


##############################
# --- (1) RUN SIMULATION --- #

# Dimension of data
n, p = 100, 150
s, k = 5, 10
b0 = +1  # Intercept
alpha = 0.1
# Number of simulations per snr
nsim = 100
n_snr = 7
frac_split_seq = [0.0, 0.25, 0.5, 0.75]
n_perm = n_snr*nsim
snr_seq_log10 = np.linspace(-1, +1, n_snr)
snr_seq = np.exp(snr_seq_log10 * np.log(10))
df_snr = pd.DataFrame({'noise':False, 'snr':snr_seq, 'beta':np.sqrt(snr_seq / s)})
df_snr = pd.concat(objs=[df_snr.assign(noise=True,beta=0),df_snr])

import warnings
warnings.simplefilter("ignore")

stime = time()
holder_sim = []
idx = 0
for j, snr in enumerate(snr_seq):
    for i in range(nsim):
        # Draw data
        idx += 1
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=idx, return_params=True)

        # (i) Loop over the difference frac to split
        for frac_split in frac_split_seq:
            split_marginal = marginal_screen(k, y, x, frac_split=frac_split)
            # Calculate the ground truth parameters
            idx_noise = ~(np.array(split_marginal.cidx_screen) < s)
            n_screened = np.sum(~idx_noise)
            sigma2_gt = 1 + (s-n_screened)*beta1[0]**2
            beta_gt = np.where(idx_noise, 0, beta1[0])
            # Do naive inference
            split_marginal.ols_screen.run_inference(alpha, 0, sigma2_gt)
            ols_naive_i = split_marginal.ols_screen.res_inf.assign(cidx=split_marginal.cidx_screen, mdl='ols')
            # Do PoSI inference
            any_split = frac_split > 0
            split_marginal.run_inference(alpha, 0, sigma2_gt, run_carve=any_split, run_split=any_split)
            tnorm_screen_i = split_marginal.res_screen.assign(cidx=split_marginal.cidx_screen, mdl='posi')
            nts_carve_i = pd.DataFrame({})
            if any_split:
                nts_carve_i = split_marginal.res_carve.assign(cidx=split_marginal.cidx_screen, mdl='carve')
            # Combine and save
            res_screen_i = pd.concat(objs=[ols_naive_i, tnorm_screen_i, nts_carve_i])
            res_screen_i = res_screen_i.assign(snr=snr, sim=i, frac_split=frac_split)
            holder_sim.append(res_screen_i)
        
        if idx % 1 == 0:
            dtime = time() - stime
            rate = idx / dtime
            seta = (n_perm - idx)/rate
            print(f'Simulation {idx} of {n_perm} (ETA={seta:0.0f} seconds)')

res_sim = pd.concat(holder_sim).reset_index(drop=True)
res_sim['noise'] = ~(res_sim['cidx'] < s)
res_sim = res_sim.merge(df_snr,'left')
res_sim = res_sim.assign(cover=lambda x: (x['lb'] <= x['beta']) & (x['ub'] >= x['beta']))
res_sim = res_sim.assign(reject=lambda x: x['pval'] < alpha)
res_sim = res_sim.assign(snr10 = lambda x: np.log10(x['snr']))


############################
# --- (2) PLOT RESULTS --- #

# Plotting libraries
import plotnine as pn
from mizani.formatters import percent_format

# Shared terms
di_mdl = {'ols':'Naive OLS', 'posi':'PoSI', 'carve':'Carving'}
di_noise1 = {'False':'True signal', 'True':'Noise'}
di_noise2 = {'False':'Type-II', 'True':'Type-I'}
cn_agg = {'mean','sum','count'}
cn_gg1 = ['noise','mdl','snr10','frac_split']
cn_gg2 = ['noise','snr10','frac_split']
cn_gg3 = ['snr10','frac_split']

# (i) Calculate coverage
res_cover = res_sim.groupby(cn_gg1)['cover'].agg(cn_agg).reset_index()
res_cover = get_CI(res_cover, cn_den='count', cn_pct='mean', alpha=alpha).rename(columns={'mean':'cover'}).drop(columns=cn_agg,errors='ignore')

gg_cover = (pn.ggplot(res_cover, pn.aes(x='snr10', y='cover', color='frac_split.astype(str)')) + 
            pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)',y='Coverage') + 
            pn.scale_color_discrete(name='Fraction used for selection') + 
            # pn.scale_color_gradient(name='Fraction used for selection') + 
            # pn.scale_shape_discrete(name='Modelling approach') + 
            pn.scale_y_continuous(labels=percent_format()) + 
            pn.geom_hline(yintercept=1-alpha, color='black', linetype='--') + 
            pn.theme(legend_position=(0.5,-0.05)) + 
            pn.facet_grid('mdl~noise',labeller=pn_labeller(noise=lambda x: di_noise1.get(x,x), mdl=lambda x: di_mdl.get(x,x))))
gg_cover.save(os.path.join(dir_figures, 'carving_cover.png'), width=7, height=6)


# (ii) Calculate type-I, type-II for PoSI models
res_type12 = res_sim.query('mdl == "posi"').groupby(cn_gg1)['reject'].agg(cn_agg).reset_index()
res_type12 = res_type12.assign(mean=lambda x: np.where(x['noise'], x['mean'], 1-x['mean']))
res_type12 = get_CI(res_type12, cn_den='count', cn_pct='mean', alpha=alpha)
res_type12 = res_type12.rename(columns={'mean':'err'}).drop(columns=cn_agg,errors='ignore')

gg_type12 = (pn.ggplot(res_type12, pn.aes(x='snr10', y='err', color='frac_split.astype(str)')) + 
            pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)', y='Error') + 
            pn.scale_color_discrete(name='Fraction used for selection') + 
            # pn.scale_color_gradient(name='Fraction used for selection') + 
            pn.scale_y_continuous(labels=percent_format()) + 
            pn.geom_hline(yintercept=alpha, color='black', linetype='--') + 
            pn.theme(legend_position=(0.5,-0.15)) + 
            pn.facet_grid('~noise',labeller=pn_labeller(noise=lambda x: di_noise2.get(x,x))))
gg_type12.save(os.path.join(dir_figures, 'carving_type12.png'), width=7, height=3.5)

# (iii) Calculate selection prob
res_sel = res_sim.query('mdl=="posi"').groupby(cn_gg3)['noise'].agg({'sum','count'}).assign(n=lambda x: x['count']-x['sum'],tot=lambda x: s*nsim).drop(columns=['count','sum']).assign(pct=lambda x: x['n']/x['tot']).reset_index()
res_sel = get_CI(res_sel, cn_den='tot', cn_num='n', alpha=alpha)
res_sel = res_sel.rename(columns={'pct':'sel_prob'}).drop(columns=cn_agg,errors='ignore')

gg_selprob = (pn.ggplot(res_sel, pn.aes(x='snr10', y='sel_prob', color='frac_split.astype(str)')) + 
            pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)', y='Selection prob') + 
            # pn.scale_color_gradient(name='Fraction used for selection') + 
            pn.scale_color_discrete(name='Fraction used for selection') + 
            pn.scale_y_continuous(labels=percent_format(),limits=[0,1]) + 
            pn.theme(legend_position=(0.5,-0.15)))
gg_selprob.save(os.path.join(dir_figures, 'carving_selection_prob.png'), width=5.5, height=3.5)










