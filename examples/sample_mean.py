"""
Shows how data carving works for simple sample mean test

python3 -m examples.sample_mean
"""

# External
import os
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm, binom
# Internal
from sntn.dists import nts, tnorm
from parameters import seed, dir_figures
from sntn.utilities.utils import rvec, cvec


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
dist_null_carve = nts(*np.broadcast_arrays(mu_null_wide, var_split, mu_null_wide, var_screen), thresh, np.inf, 1, 1)
dist_alt_carve = nts(*np.broadcast_arrays(mu_seq, var_split, mu_seq, var_screen),thresh, np.inf, 1,1)
assert np.all(dist_null_carve.ppf(0.5,method='approx')[0,:,0] == dist_alt_carve.ppf(0.5,'approx')[0,:,0]), 'expected ppfs to line up'


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
# q_carve_alt = dist_alt_carve.ppf(p_seq.flatten(), verbose_iter=1, verbose=True)
power_carve = np.mean(1 - dist_null_carve.cdf(q_carve_alt) < alpha, axis=0)
res_power_carve = pd.DataFrame(power_carve,index=n_split,columns=mu_seq.flat).rename_axis('n').melt(ignore_index=False,var_name='mu',value_name='power').reset_index().assign(method='carve')


# (iv) Combine all
res_power = pd.concat(objs=[res_power_split, res_power_screen, res_power_carve])
res_type1 = res_power.query('mu==0').reset_index(drop=True)
res_type2 = res_power.query('mu>0').reset_index(drop=True)


##############################
# --- (3) SCREENING PROB --- #



#######################
# --- (4) FIGURES --- #

import plotnine as pn
from mizani.formatters import percent_format

di_methods = {'split':'Sample-splitting', 'screen':'PoSI', 'carve':'Data carving'}

gg_power_comp = (pn.ggplot(res_type2, pn.aes(x='n', y='power', color='method')) + 
    pn.theme_bw()  + pn.geom_line(size=1) + 
    pn.labs(y='Power', x='Sample size') + 
    pn.scale_color_discrete(name='Inference',labels=lambda x: [di_methods.get(z) for z in x]) + 
    pn.geom_hline(yintercept=alpha,linetype='--') + 
    pn.scale_y_continuous(labels=percent_format()) + 
    pn.facet_wrap('~mu', labeller=pn.label_both))
gg_power_comp.save(os.path.join(dir_figures, 'power_mean_comp.png'),width=6,height=5)
