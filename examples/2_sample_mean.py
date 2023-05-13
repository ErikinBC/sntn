"""
Shows how data carving works for simple sample mean test

python3 -m examples.2_sample_mean
"""

# External
import os
import numpy as np
import pandas as pd
from time import time
from math import isclose
from scipy.stats import norm, binom
# Internal
from sntn.dists import nts, tnorm
from parameters import seed, dir_figures
from sntn.utilities.utils import rvec, cvec, pn_labeller

# Whether two arrays are close enough
tol = 1e-5


#################################
# --- (1) SAMPLE MEAN DISTS --- #

# Threshold: when do we run a test
alpha = 0.10
thresh = 1
n = 100
sigma2 = 4
sigma = np.sqrt(sigma2)
mu_null = 0
mu_seq = rvec(np.round(np.linspace(0, 1, 5),2))

# Consider the distributions over different splits
frac_split = np.round(np.arange(0.01, 1.0, 0.01), 2)
n_split = (n * frac_split).astype(int)
n_screen = n - n_split

# (i) Classical (split) distribution
var_split = cvec(sigma2 / n_split)
se_split = np.sqrt(var_split)
# Under the null, mean is always zero
dist_null_split = norm(loc=mu_null, scale=se_split)
assert dist_null_split.mean().shape == (len(frac_split),1)
# Under alternative, each alternative mean has a 1/sqrt(n) se
dist_alt_split = norm(*np.broadcast_arrays(mu_seq, se_split))
assert np.all(dist_null_split.cdf(1).flatten() == dist_alt_split.cdf(1)[:,0]), 'expected 0th column to align with null distribution'
param_shape_split = dist_alt_split.mean().shape

# (ii) PoSI: Truncated normal
var_screen = cvec(sigma2 / n_screen)
se_screen = np.sqrt(var_screen)
dist_null_screen = tnorm(mu_null, var_screen, thresh, np.inf)
dist_alt_screen = tnorm(*np.broadcast_arrays(mu_seq, var_screen),thresh,np.inf)
assert np.all(dist_null_screen.cdf(2).flatten() == dist_alt_screen.cdf(2)[:,0]), 'expected 0th column to align with null distribution'

# (iii) Data carving: NTS
mu_null_wide = np.broadcast_arrays(mu_null, mu_seq)[0]
c1 = (n_split / n).reshape(var_split.shape)
c2 = (n_screen / n).reshape(var_split.shape)
assert np.all(c1 + c2 == 1), 'weights should sum to 1'
dist_null_carve = nts(*np.broadcast_arrays(mu_null_wide, var_split, mu_null_wide, var_screen, thresh, np.inf, c1, c2), cdf_aproach='scipy')
dist_alt_carve = nts(*np.broadcast_arrays(mu_seq, var_split, mu_seq, var_screen,thresh, np.inf, c1, c2), cdf_aproach='scipy')
assert np.all(dist_null_carve.ppf(0.5,method='approx')[0,:,0] == dist_alt_carve.ppf(0.5,'approx')[0,:,0]), 'expected ppfs to line up'


###############################
# --- (2) POWER ESTIMATES --- #

# (i) Classical (split) distribution
critv_split = dist_null_split.ppf(1-alpha)
power_split = 1 - dist_alt_split.cdf(critv_split)
res_power_split = pd.DataFrame(power_split,index=n_split,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='split')
assert np.abs(res_power_split.query('mu==0')['power']-alpha).max() < tol

# (ii) PoSI: Truncated normal
critv_screen = dist_null_screen.ppf(1-alpha)
power_screen = 1 - dist_alt_screen.cdf(critv_screen)
res_power_screen = pd.DataFrame(power_screen,index=n_screen,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='screen')
assert np.abs(res_power_screen.query('mu==0')['power']-alpha).max() < tol

# (iii) Data carving: NTS
critv_carve = np.squeeze(dist_null_carve.ppf(1-alpha, verbose=True, verbose_iter=50))
power_carve = 1 - dist_alt_carve.cdf(critv_carve)
res_power_carve = pd.DataFrame(power_carve,index=n_split,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='carve')

# (iv) Combine all
res_power = pd.concat(objs=[res_power_split, res_power_screen, res_power_carve])
res_type1 = res_power.query('mu==0').reset_index(drop=True)
res_type2 = res_power.query('mu>0').reset_index(drop=True)


##############################
# --- (3) SCREENING PROB --- #

# Loss in prob of selecting "true" coordinate

# Gain in power


#######################
# --- (4) FIGURES --- #

# Figure plotting packages
import plotnine as pn
from mizani.formatters import percent_format

# Shared labels
di_methods = {'split':'Sample-splitting', 'screen':'PoSI', 'carve':'Data carving'}

# (i) How does power compare
gg_power_comp = (pn.ggplot(res_type2, pn.aes(x='n', y='power', color='method')) + 
    pn.theme_bw()  + pn.geom_line(size=1) + 
    pn.labs(y='Power', x='Samples for split') + 
    pn.scale_color_discrete(name='Inference',labels=lambda x: [di_methods.get(z) for z in x]) + 
    pn.geom_hline(yintercept=alpha,linetype='--') + 
    pn.scale_y_continuous(labels=percent_format()) + 
    pn.facet_wrap('~mu', labeller=pn.label_both))
gg_power_comp.save(os.path.join(dir_figures, 'power_mean_comp.png'),width=6,height=5)


# #####################
# # --- (MU=0.5)  --- #



# # Compare distributions (historgrams) under null and alternative for different n's
# mu_alt = 0.5
# n_range_split = np.array([45,60,75,90])
# # Why does power decline as the number of samples dedicated to splitting go up?
# idx_split = np.isin(n_split, n_range_split)
# # idx_screen = np.isin(n_screen, n_range_split)
# tau21 = var_split[idx_split].flatten()  # Variance of the classical
# tau22 = var_screen[idx_split].flatten()  # Variance of the screened truncnorm
# c1 = 1  #n_range_split/n
# c2 = 1  #1 - n_range_split/n
# dist_mu0_nts = nts(0, tau21, 0, tau22, thresh, np.inf, c1, c2, cdf_approach='scipy')
# x0 = dist_mu0_nts.mean()
# dist_mu5_nts = nts(mu_alt, tau21, mu_alt, tau22, thresh, np.inf, c1, c2)

# # Draw random data
# nsamp = 1000
# dat_mu0_nts = pd.DataFrame(dist_mu0_nts.rvs(nsamp, seed=seed),columns=n_range_split)
# dat_mu0_nts = dat_mu0_nts.melt(var_name='n').assign(dist='null')
# dat_mu5_nts = pd.DataFrame(dist_mu5_nts.rvs(nsamp, seed=seed),columns=n_range_split)
# dat_mu5_nts = dat_mu5_nts.melt(var_name='n').assign(dist='alt')
# dat_mu05_nts = pd.concat(objs=[dat_mu0_nts, dat_mu5_nts]).reset_index(drop=True)
# # Calculate the 1-alpha quantile for the null
# dat_mu0_critv = pd.DataFrame({'n':n_range_split,'critv':dist_mu0_nts.ppf(1-alpha).flat})
# dat_mu0_critv = dat_mu0_critv.merge(dat_mu5_nts.merge(dat_mu0_critv).assign(reject=lambda x: x['value']>x['critv']).groupby('n')['reject'].mean().reset_index())

# # Plot the respective distributions
# di_dist = {'null':'Null (mu=0)', 'alt':f'Alt (mu={mu_alt})'}
# gg_mu05_nts1 = (pn.ggplot(dat_mu05_nts, pn.aes(x='value',fill='dist')) + pn.theme_bw() + 
#             pn.geom_density(color='black',alpha=0.5) + 
#             pn.labs(x='Statistic',y='Frequency') + 
#             pn.facet_wrap('~n',labeller=pn.label_both) + 
#             pn.geom_vline(pn.aes(xintercept='critv'),data=dat_mu0_critv,color='#00BFC4',size=1,linetype='--') + 
#             pn.geom_text(pn.aes(label='100*reject',x='critv',y=1.5),size=10,color='#00BFC4',nudge_x=0.5,format_string='{:.0f}%',data=dat_mu0_critv,inherit_aes=False) +
#             pn.scale_fill_discrete(name='Distribution',labels=lambda x: [di_dist.get(z) for z in x]))
# gg_mu05_nts1.save(os.path.join(dir_figures, 'dist_nts_null_alt1.png'),width=7,height=6)

# gg_mu05_nts2 = (pn.ggplot(dat_mu05_nts, pn.aes(x='value',fill='n.astype(str)')) + pn.theme_bw() + 
#             pn.geom_density(color='black',alpha=0.5) + 
#             pn.labs(x='Statistic',y='Frequency') + 
#             pn.facet_wrap('~dist',labeller=pn_labeller(dist=lambda x: di_dist.get(x,x))) + 
#             # pn.geom_vline(pn.aes(xintercept='critv'),data=dat_mu0_critv,color='#00BFC4',size=1,linetype='--') + 
#             pn.scale_fill_discrete(name='Split (n)'))
# gg_mu05_nts2.save(os.path.join(dir_figures, 'dist_nts_null_alt2.png'),width=7,height=4)

