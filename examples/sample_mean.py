"""
Shows how data carving works for simple sample mean test

python3 -m examples.sample_mean
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

#############################
# --- (0) SANITY CHECK --- #


# In the limit, when n=99, we have a very tight distribution around our means ±0.2, so power is effectively 90% for mu=0.5. The null of the truncated normal is a tnorm(0,sigma2,1,inf), which has a very large value
alpha = 0.1
nsim = 1000000
a = 1
b = np.inf
mu_null = 0.0
mu_alt = 0.5
sigma2 = 4
n = 100
n_split = 75
n_screen = 100 - n_split
sd_split = np.sqrt(sigma2 / n_split)
sd_screen = np.sqrt(sigma2 / n_screen)
c1 = n_split / n
c2 = n_screen / n

# Use NTS class to generate critival values and cdf (power)
nts_null_u = nts(mu_null, sd_split**2, mu_null, sd_screen**2, a, b, 1, 1)
nts_null_w = nts(mu_null, sd_split**2, 0, sd_screen**2, a, b, c1, c2)
nts_alt_u = nts(mu_alt, sd_split**2, mu_alt, sd_screen**2, a, b, 1, 1)
nts_alt_w = nts(mu_alt, sd_split**2, mu_alt, sd_screen**2, a, b, c1, c2)
# Critical values
crit_v_u = nts_null_u.ppf(1-alpha)[0]
# crit_v_w = nts_null_w.ppf(1-alpha) # FAILS
power_u = 1-nts_alt_u.cdf(crit_v_u)[0]
power_w = 1
# power_w = 1-nts_alt_w.cdf(crit_v_w)[0]
assert np.abs(power_u - np.mean(nts_alt_u.rvs(nsim, seed=seed) > crit_v_u)) < 1e-4, 'cdf != rvs'

# Repeat with rvs
rvs_null_u = 1*norm(0, sd_split).rvs(nsim,random_state=seed) + 1*tnorm(0,sd_screen**2, a, b).rvs(nsim,seed=seed)
rvs_null_w = c1*norm(0, sd_split).rvs(nsim,random_state=seed) + c2*tnorm(0,sd_screen**2, a, b).rvs(nsim,seed=seed)
rvs_alt_u = 1*norm(mu_alt, sd_split).rvs(nsim,random_state=seed) + 1*tnorm(mu_alt,sd_screen**2, a, b).rvs(nsim,seed=seed)
rvs_alt_w = c1*norm(mu_alt, sd_split).rvs(nsim,random_state=seed) + c2*tnorm(mu_alt,sd_screen**2, a, b).rvs(nsim,seed=seed)
emp_critv_u = np.quantile(rvs_null_u, 1-alpha)
emp_power_u = np.mean(rvs_alt_u > emp_critv_u)
emp_critv_w = np.quantile(rvs_null_w, 1-alpha)
emp_power_w = np.mean(rvs_alt_w > emp_critv_w)

# Compare results
print(f'~~~ c1={c1:.2f} (n1={n_split}), c2={c2:.2f} (n2={n_screen}) ~~~')
print(f'Unweighted power for theory={power_u*100:.2f}%, empirical={emp_power_u*100:.2f}%')
print(f'Weighted power for theory={power_w*100:.2f}%, empirical={emp_power_w*100:.2f}%')





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
dist_null_carve = nts(*np.broadcast_arrays(mu_null_wide, var_split, mu_null_wide, var_screen, thresh, np.inf, c1, c2))
dist_alt_carve = nts(*np.broadcast_arrays(mu_seq, var_split, mu_seq, var_screen,thresh, np.inf, c1, c2))
assert np.all(dist_null_carve.ppf(0.5,method='approx')[0,:,0] == dist_alt_carve.ppf(0.5,'approx')[0,:,0]), 'expected ppfs to line up'


#####################
# --- (MU=0.5)  --- #

import plotnine as pn
from mizani.formatters import percent_format


# Compare distributions (historgrams) under null and alternative for different n's
mu_alt = 0.5
n_range_split = np.array([45,60,75,90])
# Why does power decline as the number of samples dedicated to splitting go up?
idx_split = np.isin(n_split, n_range_split)
# idx_screen = np.isin(n_screen, n_range_split)
tau21 = var_split[idx_split].flatten()  # Variance of the classical
tau22 = var_screen[idx_split].flatten()  # Variance of the screened truncnorm
c1 = 1  #n_range_split/n
c2 = 1  #1 - n_range_split/n
dist_mu0_nts = nts(0, tau21, 0, tau22, thresh, np.inf, c1, c2, cdf_approach='scipy')
x0 = dist_mu0_nts.mean()
dist_mu5_nts = nts(mu_alt, tau21, mu_alt, tau22, thresh, np.inf, c1, c2)

# Draw random data
nsamp = 1000
dat_mu0_nts = pd.DataFrame(dist_mu0_nts.rvs(nsamp, seed=seed),columns=n_range_split)
dat_mu0_nts = dat_mu0_nts.melt(var_name='n').assign(dist='null')
dat_mu5_nts = pd.DataFrame(dist_mu5_nts.rvs(nsamp, seed=seed),columns=n_range_split)
dat_mu5_nts = dat_mu5_nts.melt(var_name='n').assign(dist='alt')
dat_mu05_nts = pd.concat(objs=[dat_mu0_nts, dat_mu5_nts]).reset_index(drop=True)
# Calculate the 1-alpha quantile for the null
dat_mu0_critv = pd.DataFrame({'n':n_range_split,'critv':dist_mu0_nts.ppf(1-alpha).flat})
dat_mu0_critv = dat_mu0_critv.merge(dat_mu5_nts.merge(dat_mu0_critv).assign(reject=lambda x: x['value']>x['critv']).groupby('n')['reject'].mean().reset_index())

# Plot the respective distributions
di_dist = {'null':'Null (mu=0)', 'alt':f'Alt (mu={mu_alt})'}
gg_mu05_nts1 = (pn.ggplot(dat_mu05_nts, pn.aes(x='value',fill='dist')) + pn.theme_bw() + 
            pn.geom_density(color='black',alpha=0.5) + 
            pn.labs(x='Statistic',y='Frequency') + 
            pn.facet_wrap('~n',labeller=pn.label_both) + 
            pn.geom_vline(pn.aes(xintercept='critv'),data=dat_mu0_critv,color='#00BFC4',size=1,linetype='--') + 
            pn.geom_text(pn.aes(label='100*reject',x='critv',y=1.5),size=10,color='#00BFC4',nudge_x=0.5,format_string='{:.0f}%',data=dat_mu0_critv,inherit_aes=False) +
            pn.scale_fill_discrete(name='Distribution',labels=lambda x: [di_dist.get(z) for z in x]))
gg_mu05_nts1.save(os.path.join(dir_figures, 'dist_nts_null_alt1.png'),width=7,height=6)

gg_mu05_nts2 = (pn.ggplot(dat_mu05_nts, pn.aes(x='value',fill='n.astype(str)')) + pn.theme_bw() + 
            pn.geom_density(color='black',alpha=0.5) + 
            pn.labs(x='Statistic',y='Frequency') + 
            pn.facet_wrap('~dist',labeller=pn_labeller(dist=lambda x: di_dist.get(x,x))) + 
            # pn.geom_vline(pn.aes(xintercept='critv'),data=dat_mu0_critv,color='#00BFC4',size=1,linetype='--') + 
            pn.scale_fill_discrete(name='Split (n)'))
gg_mu05_nts2.save(os.path.join(dir_figures, 'dist_nts_null_alt2.png'),width=7,height=4)




###############################
# --- (2) POWER ESTIMATES --- #

# One-sided test (use the quantiles to approximate the power)
p_seq = np.arange(0.001,1,0.001)
p_seq = np.expand_dims(p_seq, [1,2])

# (i) Classical (split) distribution
q_split_alt = dist_alt_split.ppf(p_seq)
power_split = np.mean(1-dist_null_split.cdf(q_split_alt) < alpha, axis=0)
res_power_split = pd.DataFrame(power_split,index=n_split,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='split')

# (ii) PoSI: Truncated normal
q_screen_alt = dist_alt_screen.ppf(p_seq)
power_screen = np.mean(1 - dist_null_screen.cdf(q_screen_alt) < alpha, axis=0)
res_power_screen = pd.DataFrame(power_screen,index=n_screen,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='screen')

# (iii) Data carving: NTS
# Solving thousands of roots takes to long, use empirical quantiles
ndraw = 25000
q_carve_alt = np.quantile(dist_alt_carve.rvs(ndraw, seed=seed),p_seq.flat, axis=0)
# FAILS
# q_carve_alt = dist_alt_carve.ppf(p_seq.flatten(), verbose_iter=1, verbose=True)
power_carve = np.mean(1 - dist_null_carve.cdf(q_carve_alt) < alpha, axis=0)
res_power_carve = pd.DataFrame(power_carve,index=n_split,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='carve')

# (iv) Combine all
res_power = pd.concat(objs=[res_power_split, res_power_screen, res_power_carve])
res_type1 = res_power.query('mu==0').reset_index(drop=True)
res_type2 = res_power.query('mu>0').reset_index(drop=True)






[idx_screen]

# Draw data for histograms for the null, and calculate rejection threshold


##############################
# --- (3) SCREENING PROB --- #



#######################
# --- (4) FIGURES --- #


di_methods = {'split':'Sample-splitting', 'screen':'PoSI', 'carve':'Data carving'}

gg_power_comp = (pn.ggplot(res_type2, pn.aes(x='n', y='power', color='method')) + 
    pn.theme_bw()  + pn.geom_line(size=1) + 
    pn.labs(y='Power', x='Sample size') + 
    pn.scale_color_discrete(name='Inference',labels=lambda x: [di_methods.get(z) for z in x]) + 
    pn.geom_hline(yintercept=alpha,linetype='--') + 
    pn.scale_y_continuous(labels=percent_format()) + 
    pn.facet_wrap('~mu', labeller=pn.label_both))
gg_power_comp.save(os.path.join(dir_figures, 'power_mean_comp.png'),width=6,height=5)
