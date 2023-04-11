"""
Simulations (analyses) for the tnorm class:

i) Which solvers should be considered for further testing?
ii) Which solvers get the right coverage and which is the fastest?
iii) Does the built in "fit" method work?

python3 -m simulations.tnorm
"""

# External
import os
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
# Internal
from sntn.dists import tnorm
from sntn.utilities.utils import pseudo_log10
from parameters import dir_figures, dir_simulations
# For multiindex slicing
idx = pd.IndexSlice


################################
# ---- (i) SCREEN SOLVERS ---- #

# Answer: Loading the unittests, we find...

# Load the data generated from test_dist_tnorm.py
di_bound = {'lb':'Lower-bound', 'ub':'Upper-bound'}
res_solver_utests = pd.read_csv(os.path.join(dir_simulations, 'res_test_norm_CI.csv'))

# res_all['idx'] = pd.Categorical(res_all['idx'] + 1, range(1,ndraw+1))
# res_all = res_all.assign(num=lambda x: (pd.Categorical(x['method']).codes+1))
# res_all = res_all.assign(color = lambda x: x['num'].astype(str) + '.' + x['method'])
# # Add on the true parameters
# assert len(mu) == 1, 'Cannot assign if value > 1'


mapdi = res_solver_utests.groupby('num').apply(lambda x: x['color'].unique()[0]).to_dict()
mu0 = res_solver_utests['mu0'].unique()[0]
# Screen for only "reasonable" candidates
wide_solver_utests = res_solver_utests.pivot(index=['idx','method','approach'],columns='bound',values='value')
wide_solver_utests = wide_solver_utests.assign(width=lambda x: x['ub']-x['lb'])
wide_solver_utests = wide_solver_utests.assign(cover=lambda x: (x['lb'] <= mu0) & (x['ub'] >= mu0))
wide_solver_freq = wide_solver_utests.groupby(['method','approach'])[['cover','width']].mean()
best_solvers = wide_solver_freq.round(3).reset_index().merge(wide_solver_freq.round(3).groupby(['cover','width']).size().sort_values(ascending=False).head(1).reset_index())[['method','approach']]
print(f'Out of {len(wide_solver_freq)} solvers, a total of {len(best_solvers)} show stable estimates')





##########################################
# ---- (ii) SOLVER COVERAGE/RUNTIME ---- #

# Answer: ....


###################################
# ---- (iii) SCIPY.DISTS.FIT ---- #

# Answer: Does not work well

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

