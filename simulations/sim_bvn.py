"""
Check speed/accuracy of different BVN approaches, each approach will be benchmarked to 10,000,000 randomly generated values
"""

# External
import os
import numpy as np
import pandas as pd
from time import time
from scipy.stats import multivariate_normal as mvn
# Internal
from sntn.dists import bvn
from sntn._bvn import valid_cdf_approach
from tests.test_dists_bvn import gen_params
from parameters import dir_figures, dir_simulations, seed

# Pre-define path
path_sim_bvn = os.path.join(dir_simulations, 'bvn_runtime.csv')


###############################
# --- (1) RUN SIMULATIONS --- #

# Number of draws to determine ground truth CDF
ndraw = 10000000
# Number of random parameters to assess accuracy
nsamp = 25
# Number of samples to test the run-time on
n = 1000
params_shape = [(n,), (n, 2), (n, 3), (n, 2, 2), (n, 3, 3), (n, 2, 2, 2), (n, 3, 3, 3), ]
n_iter = len(params_shape)

holder_shape = []
for jj, shape in enumerate(params_shape):
    print(f'--- Shape {jj+1} of {n_iter}: {shape} ---')
    # Draw parameters
    n_params = int(np.prod(shape))
    mu1, sigma21, mu2, sigma22, rho = gen_params(shape, seed)
    # Draw data
    dgp = bvn(mu1, sigma21, mu2, sigma22, rho)
    X = np.squeeze(dgp.rvs(1, seed))
    x1, x2 = np.take(X, 0, -1), np.take(X, 1, -1)
    # Draw data to determine ground truth CDF...
    np.random.seed(seed)
    idx_samp = np.random.choice(n_params, nsamp)
    holder_gt = []
    for kk, ii in enumerate(np.ndindex(shape)):
        if kk in idx_samp:
            print(f'- Drawing sample {kk} -')
            x1_ii, x2_ii = x1[ii], x2[ii]
            mu1_ii, sigma21_ii, mu2_ii, sigma22_ii, rho_ii = mu1[ii], sigma21[ii], mu2[ii], sigma22[ii], rho[ii]
            Xdraw_ii = bvn(mu1_ii, sigma21_ii, mu2_ii, sigma22_ii, rho_ii).rvs(ndraw)
            cdf_ii = np.mean( (Xdraw_ii[:,0,0] <= x1_ii) & (Xdraw_ii[:,0,1] <= x2_ii) )
            val_ii = pd.DataFrame({'x1':x1_ii, 'x2':x2_ii, 'cdf':cdf_ii, 'idx':str(ii)},index=[kk])
            holder_gt.append(val_ii)
    df_gt = pd.concat(holder_gt).assign(shape=str(shape)).reset_index(drop=True).rename_axis('samp').reset_index()
    
    # Get CDF estimate for each approach
    holder_approach = []
    for approach in valid_cdf_approach:
        print(f'- Running for approach {approach} -')
        dist = bvn(mu1, sigma21, mu2, sigma22, rho, approach)
        stime = time()
        pval = dist.cdf(x1=x1, x2=x2)
        dtime = time() - stime
        tmp = pd.DataFrame({'approach':approach, 'dtime':dtime, 'pval':df_gt['idx'].apply(lambda x: pval[eval(x)])})
        holder_approach.append(tmp)
    # Merge and compare to ground truth
    df_dtime = pd.concat(holder_approach).rename_axis('samp').reset_index()
    df_shape = df_gt.merge(df_dtime,'left','samp')
    holder_shape.append(df_shape)
# Combine all results and save
res_bvn = pd.concat(holder_shape).reset_index(drop=True)
res_bvn.to_csv(path_sim_bvn, index=False)


############################
# --- (2) PLOT RESULTS --- #

# Load plotting
import plotnine as pn
# Evaluation grouping
cn_gg = ['approach','n_params']

# (i) Clean up columns
dat_bvn = res_bvn.drop(columns='idx').assign(n_params=lambda x: x['shape'].apply(lambda x: int(np.prod(eval(x))))).drop(columns='shape')
dat_bvn = dat_bvn.assign(log_err=lambda x: -np.log10(np.abs(x['pval']-x['cdf'])))

# (ii) Get the average runtime as a function of the number of calculations
assert np.all(dat_bvn.groupby(cn_gg)['dtime'].var() == 0), 'expected dtime to be the same'
dat_dtime = dat_bvn.groupby(cn_gg)['dtime'].mean().reset_index()
dat_dtime = dat_dtime.assign(rate=lambda x: (x['n_params']/1000) / x['dtime'] )
rate_dtime = dat_dtime.drop(columns='rate').groupby('approach').sum().assign(rate=lambda x: (x['n_params']/1000) / x['dtime'] ).reset_index()

# (iii) Get cumulative error
dat_err = dat_bvn.groupby(cn_gg)['log_err'].sum().groupby('approach').cumsum().reset_index()

# (iv) Plot trade-off (total err vs average rate)
dat_tradeoff = rate_dtime.drop(columns='n_params').merge(dat_err.groupby('approach').tail(1).drop(columns='n_params'),'left','approach')
gg_bvn_tradeoff = (pn.ggplot(dat_tradeoff, pn.aes(x='rate', y='log_err', color='approach')) + 
    pn.theme_bw() + pn.scale_x_log10() + 
    pn.ggtitle('BVN tade-off by approach') +
    pn.scale_color_discrete(name='BVN integration approach') + 
    pn.labs(x='# of calculations per 1K params', y='-log10(error)') + 
    pn.geom_point(size=2))
gg_bvn_tradeoff.save(os.path.join(dir_figures,'bvn_tradeoff.png'),width=5.5,height=3.5)

# (v) Plot the cumulative error vs the runtime
dat_scaling = dat_dtime.merge(dat_err).drop(columns='dtime').assign(err2param=lambda x: x['log_err']/x['n_params'])
u_params = dat_scaling['n_params'].unique()
dat_scaling['n_params'] = pd.Categorical(dat_scaling['n_params'], u_params)
lbls = [f'${x+1}$' for x in range(len(u_params))]

gg_bvn_scaling = (pn.ggplot(dat_scaling, pn.aes(x='rate',y='err2param', color='approach', shape='n_params',group='approach')) + 
    pn.theme_bw() + pn.scale_x_log10() + 
    pn.geom_point(size=2) + pn.geom_line() + 
    pn.labs(x='# of calculations per 1K params', y='-log10(error)/param') + 
    pn.scale_shape_manual(name='Number of parameters',values=lbls) + 
    pn.scale_color_discrete(name='BVN integration approach') + 
    pn.ggtitle('BVN scaling'))
gg_bvn_scaling.save(os.path.join(dir_figures,'bvn_scaling.png'),width=5.5,height=3.5)
