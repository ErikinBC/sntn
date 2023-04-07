"""
Shows that that ML/MoM estimator does poorly when mu is outside of the support

python3 -m simulations.tnorm_fit
"""

# External
import os
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
# Internal
from parameters import dir_figures
from sntn.dists import tnorm

# Set up simulation parameters
nsim = 25
a = -1
b = +1
ndraw_seq = [10, 100, 1000, 10000]
sigma_seq = [0.1, 1, 3]
mu_seq = np.arange(-3, 3+1)
use_sigma_seq = [True,False]
ncomb = len(ndraw_seq) * len(sigma_seq) * len(mu_seq) * len(use_sigma_seq)

# Loop over all combinations
i = 0
holder = []
stime = time()
for mu in mu_seq:
    for sigma in sigma_seq:
        for ndraw in ndraw_seq:
            for use_sigma in use_sigma_seq:
                i += 1
                print(f'Iteration {i} of {ncomb}')
                mu_vec = np.repeat(mu, nsim)
                dist = tnorm(mu=mu_vec, sigma2=sigma**2, a=a, b=b)
                x = dist.rvs(ndraw, seed=i)
                mu_hat = dist.fit(x, use_a=True, use_b=True, use_sigma=use_sigma)[2]
                tmp = pd.DataFrame({'mu_hat':mu_hat, 'mu_act':mu_vec, 'sigma':sigma, 'n':ndraw, 'fix_sigma':use_sigma})
                holder.append(tmp)
                # Get ETA
                dtime, nleft = time() - stime, ncomb - i
                rate = (i+1) / dtime
                meta = (nleft / rate) / 60
                print(f'ETA = {meta:.1f} minutes remaining')
# Merge
res_fit = pd.concat(holder).reset_index(drop=True)
res_fit['mu_act'] = res_fit['mu_act'].astype(int)
res_fit = res_fit.assign(x=lambda x: pd.Categorical(x['mu_act'],x['mu_act'].unique()))
res_fit = res_fit.assign(err=lambda x: x['mu_act'] - x['mu_hat'])
dat_vlines = res_fit[['x']].query('x.isin([@a,@b])').groupby('x').size().reset_index().rename(columns={0:'n'}).query('n>0')

# Plot scatter
posd = pn.position_dodge(0.5)
gtit = 'Dashed black line shows x==y\n Blue vertical lines show trunctation range'
gg_tnorm_fit = (pn.ggplot(res_fit, pn.aes(x='mu_act',y='err',color='n.astype(str)',shape='sigma.astype(str)')) + 
    pn.labs(x='Actual mean',y='Actual less estimated mean') + 
    pn.theme_bw() + pn.ggtitle(gtit) + 
    pn.geom_point(position=posd,size=0.5,alpha=0.25) + 
    # pn.geom_boxplot() + 
    # pn.geom_violin(position=posd) + 
    # pn.geom_abline(slope=1,intercept=0,linetype='--') +
    # pn.geom_vline(pn.aes(xintercept='x'),data=dat_vlines,color='blue') + 
    pn.geom_vline(xintercept=[a,b],color='blue',linetype='--') + 
    pn.scale_color_discrete(name='Sample size') + 
    pn.scale_shape_discrete(name='Standard dev') + 
    pn.scale_x_continuous(breaks=mu_seq) + 
    pn.facet_grid('sigma~fix_sigma',labeller=pn.label_both))
gg_tnorm_fit.save(os.path.join(dir_figures, 'tnorm_fit.png'),width=7, height=6)

