"""
Show that marginal screening works, and what the power is

python3 -m examples.3_marginal_screening
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

# Suppress root finding warnings
import warnings
warnings.simplefilter("ignore")

# Set up simulation parameters
# Dimension of data
n, p = 100, 150
s, k = 5, 10
b0 = +1  # Intercept
alpha = 0.1
# Number of simulations per snr
nsim = 500
n_snr = 7
frac_split_seq = [0.15, 0.20, 0.25]
n_perm = n_snr*nsim
snr_seq_log10 = np.linspace(-1, +1, n_snr)
snr_seq = np.exp(snr_seq_log10 * np.log(10))
df_snr = pd.DataFrame({'noise':False, 'snr':snr_seq, 'beta':np.sqrt(snr_seq / s)})
df_snr = pd.concat(objs=[df_snr.assign(noise=True,beta=0),df_snr])


##############################
# --- (1) RUN SIMULATION --- #

stime = time()
holder_sim = []
idx = 0
for j, snr in enumerate(snr_seq):
    for i in range(nsim):
        # Draw data
        idx += 1
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, seed=idx+3500, return_params=True)
        # Loop over the difference frac to split
        for frac_split in frac_split_seq:
            # (i) Split x into screening and split
            split_marginal = marginal_screen(k, y, x, frac_split=frac_split)
            
            # (ii) Calculate the ground truth parameters on "selected" model
            idx_noise = ~(np.array(split_marginal.cidx_screen) < s)
            n_screened = np.sum(~idx_noise)
            sigma2_gt = 1 + (s-n_screened)*beta1[0]**2
            beta_gt = np.where(idx_noise, 0, beta1[0])
            
            # (iii) Do naive inference (type-I errors shuold be inflated)
            split_marginal.ols_screen.run_inference(alpha, 0, sigma2_gt)
            ols_naive_i = split_marginal.ols_screen.res_inf.assign(cidx=split_marginal.cidx_screen, mdl='naive')
            ols_naive_i = ols_naive_i[['bhat','pval','cidx','mdl']]

            # (iv) Run PoSI inference for screened data
            any_split = frac_split > 0
            split_marginal.run_inference(alpha, 0, sigma2_gt, run_carve=any_split, run_split=any_split, run_ci=False)
            tnorm_screen_i = split_marginal.res_screen.assign(cidx=split_marginal.cidx_screen, mdl='screen')
            
            nts_carve_i, gauss_split_i = pd.DataFrame({}), pd.DataFrame({})
            if any_split:
                gauss_split_i = split_marginal.res_split.drop(columns=['lb','ub']).assign(mdl='split')
                nts_carve_i = split_marginal.res_carve.assign(cidx=split_marginal.cidx_screen, mdl='carve')
            # Combine and save
            res_screen_i = pd.concat(objs=[ols_naive_i, tnorm_screen_i, gauss_split_i, nts_carve_i])
            res_screen_i = res_screen_i.assign(snr=snr, sim=i, frac_split=frac_split)
            holder_sim.append(res_screen_i)
        
        if idx % 25 == 0:
            dtime = time() - stime
            rate = idx / dtime
            seta = (n_perm - idx)/rate
            print(f'Simulation {idx} of {n_perm} (ETA={seta:0.0f} seconds)')
res_sim = pd.concat(holder_sim).reset_index(drop=True).drop(columns='se',errors='ignore')
res_sim['noise'] = ~(res_sim['cidx'] < s)
res_sim = res_sim.merge(df_snr,'left')
# res_sim = res_sim.assign(cover=lambda x: (x['lb'] <= x['beta']) & (x['ub'] >= x['beta']))
res_sim = res_sim.assign(reject=lambda x: x['pval'] < alpha)
res_sim = res_sim.assign(snr10 = lambda x: np.log10(x['snr']))
res_sim.to_csv(os.path.join('examples','marginal_screening_sim.csv'),index=False)
if 'res_sim' not in dir():
    res_sim = pd.read_csv(os.path.join('examples','marginal_screening_sim.csv'))


############################
# --- (2) PLOT RESULTS --- #

# Plotting libraries
import plotnine as pn
from mizani.formatters import percent_format

# Shared terms
di_mdl = {'naive':'Naive OLS', 'carve':'Data carving', 'screen':'PoSI', 'split':'Sample splitting'}
di_noise1 = {'False':'True signal', 'True':'Noise'}
di_noise2 = {'False':'Type-II', 'True':'Type-I'}
di_frac = dict(zip([str(frac) for frac in frac_split_seq],[f'Split={100*frac:0.0f}%' for frac in frac_split_seq]))
cn_agg = {'mean','sum','count'}
cn_gg1 = ['noise','mdl','snr10','frac_split']
cn_gg2 = ['noise','snr10','frac_split']
cn_gg3 = ['snr10','frac_split']

# (i) Calculate type-I/II error
res_type12 = res_sim.groupby(cn_gg1)['reject'].agg(cn_agg).reset_index()
res_type12.rename(columns={'mean':'err'}, inplace=True)
# If covariate is noise, keep type-1 error, if not, convert power to type-II error
res_type12 = res_type12.assign(err=lambda x: np.where(x['noise'], x['err'], 1-x['err']))
res_type12 = get_CI(res_type12, cn_den='count', cn_pct='err', alpha=alpha)
res_type12.drop(columns=cn_agg,errors='ignore', inplace=True)

# (ii) Plot the type-I error
dat_type1 = res_type12.query('noise').drop(columns='noise').reset_index(drop=True)
dat_type1['mdl'] = pd.Categorical(dat_type1['mdl'],list(di_mdl))
dat_type2 = res_type12.query('~noise & mdl!="naive"').drop(columns='noise').reset_index(drop=True)
dat_type2['mdl'] = pd.Categorical(dat_type2['mdl'],list(di_mdl)).remove_unused_categories()

colz1 = ["black", "#F8766D", "#00BA38", "#619CFF"]
colz2 = colz1[1:]

gg_type1 = (pn.ggplot(dat_type1, pn.aes(x='snr10', y='err', color='mdl')) + 
            pn.theme_bw() + 
            pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)', y='Type-I Error') + 
            pn.scale_color_manual(name='Inference', values=colz1, labels=lambda x: [di_mdl.get(z) for z in x]) + 
            pn.scale_y_continuous(labels=percent_format()) + 
            pn.geom_hline(yintercept=alpha, color='black', linetype='--') + 
            pn.theme(legend_position=(0.5,-0.1)) + 
            pn.facet_wrap('~frac_split',labeller=pn_labeller(frac_split=lambda x: di_frac.get(x,x))))
gg_type1.save(os.path.join(dir_figures, 'marginal_screening_type1.png'), width=9, height=3)

# Repeat for type2
gg_type2 = (pn.ggplot(dat_type2, pn.aes(x='snr10', y='err', color='mdl',linetype='frac_split.astype(str)')) + 
            pn.theme_bw() + 
            pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)', y='Type-2 Error') + 
            pn.scale_color_manual(name='Inference', values=colz2,labels=lambda x: [di_mdl.get(z) for z in x]) + 
            pn.scale_linetype_discrete(name='Fraction',labels=lambda x: [di_frac.get(z) for z in x] ) + 
            pn.scale_y_continuous(labels=percent_format(),limits=[0,1]) + 
            pn.theme(legend_position=(0.5,-0.10)))
            # pn.facet_wrap('~frac_split',labeller=pn_labeller(frac_split=lambda x: di_frac.get(x,x))))
gg_type2.save(os.path.join(dir_figures, 'marginal_screening_type2.png'), width=5, height=4)



# (ii) Calculate selection prob
res_sel = res_sim.query('mdl=="screen"').groupby(cn_gg3)['noise'].agg({'sum','count'}).assign(n=lambda x: x['count']-x['sum'],tot=lambda x: s*nsim).drop(columns=['count','sum']).assign(pct=lambda x: x['n']/x['tot']).reset_index()
res_sel = get_CI(res_sel, cn_den='tot', cn_num='n', alpha=alpha)
res_sel = res_sel.rename(columns={'pct':'sel_prob'}).drop(columns=cn_agg,errors='ignore')
gg_selprob = (pn.ggplot(res_sel, pn.aes(x='snr10', y='sel_prob', color='frac_split.astype(str)')) + 
            pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
            pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
            pn.labs(x='log10(SNR)', y='Selection prob') + 
            pn.scale_color_discrete(name='Fraction',labels=lambda x: [di_frac.get(z) for z in x] ) + 
            pn.scale_y_continuous(labels=percent_format(),limits=[0,1]) + 
            pn.theme(legend_position=(0.5,-0.17)))
gg_selprob.save(os.path.join(dir_figures, 'marginal_screening_selprob.png'), width=4.5, height=3)


# # (iii) Calculate coverage from CIs
# res_cover = res_sim.groupby(cn_gg1)['cover'].agg(cn_agg).reset_index()
# res_cover = get_CI(res_cover, cn_den='count', cn_pct='mean', alpha=alpha).rename(columns={'mean':'cover'}).drop(columns=cn_agg,errors='ignore')

# gg_cover = (pn.ggplot(res_cover, pn.aes(x='snr10', y='cover', color='frac_split.astype(str)')) + 
#             pn.theme_bw() + pn.geom_point() + pn.geom_line() + 
#             pn.geom_linerange(pn.aes(ymin='lb',ymax='ub')) + 
#             pn.labs(x='log10(SNR)',y='Coverage') + 
#             pn.scale_color_discrete(name='Fraction used for selection') + 
#             # pn.scale_color_gradient(name='Fraction used for selection') + 
#             # pn.scale_shape_discrete(name='Modelling approach') + 
#             pn.scale_y_continuous(labels=percent_format()) + 
#             pn.geom_hline(yintercept=1-alpha, color='black', linetype='--') + 
#             pn.theme(legend_position=(0.5,-0.05)) + 
#             pn.facet_grid('mdl~noise',labeller=pn_labeller(noise=lambda x: di_noise1.get(x,x), mdl=lambda x: di_mdl.get(x,x))))
# gg_cover.save(os.path.join(dir_figures, 'carving_cover.png'), width=7, height=6)









