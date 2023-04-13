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
from sntn.utilities.utils import pseudo_log10, mean_total_error, cat_by_order
from parameters import dir_figures, dir_simulations
# For multiindex slicing
idx = pd.IndexSlice

# Hard-coded terms
alpha = 0.05
di_bound = {'lb':'Lower-bound', 'ub':'Upper-bound'}
cn_eval = ['approach','method']  # What we are assessing
cn_idx = ['n','ndraw','idx']  # For pivoting


################################
# ---- (i) SCREEN SOLVERS ---- #

# Answer: Loading the unittests, we find...

# --- Load & process unittest data --- #
fn_test_norm = pd.Series(os.listdir(dir_simulations))
fn_test_norm = fn_test_norm[fn_test_norm.str.contains('res_test_norm',regex=False)]
fn_test_norm.reset_index(drop=True,inplace=True)
holder = []
for fn in fn_test_norm:
    holder.append(pd.read_csv(os.path.join(dir_simulations, fn)))
res_solver_utests = pd.concat(holder).reset_index(drop=True)
approach_method_count = res_solver_utests.groupby(cn_eval).size()
assert approach_method_count.var() == 0, 'Each approach/method should have same number of obs'
# Check for n/ndraw consistency
assert (res_solver_utests.groupby(['n','ndraw']).size() / int(approach_method_count.shape[0] * 2)).astype(int).reset_index().rename(columns={0:'tot'}).assign(num=lambda x: x['ndraw']*x['n'].apply(lambda z: np.prod(eval(z)))).assign(check=lambda x: x['tot']==x['num'])['check'].all(), 'Expected number of n/ndraw to align with # of approach/methods (times 2 for lb/ub)'
ci, x, mu, sigma2, a, b = [res_solver_utests[cn] for cn in ['value','x','mu','sigma2','a','b']]
# Under its own estimate, how close with the ci-mu to getting the right alpha?
dist_ci = tnorm(ci, sigma2, a, b)
pval_ci = dist_ci.cdf(x)
res_solver_utests = res_solver_utests.assign(pval_ci=pval_ci)
res_solver_utests = res_solver_utests.assign(pval_err=lambda x:  x['pval_ci']-(np.where(x['bound']=='lb',1-alpha/2,alpha/2)))

# --- Calculate the p-value error and coverage --- #
err_val = res_solver_utests.groupby(cn_eval)['pval_err'].agg(mean_total_error).reset_index()
err_val = cat_by_order(err_val, 'pval_err', 'method')

gg_err_tnorm_utests = (pn.ggplot(err_val, pn.aes(x='method',y='-np.log10(pval_err)',color='approach')) + 
    pn.theme_bw() + pn.geom_point(size=2) + 
    pn.scale_color_discrete(name='scipy approach') + 
    pn.labs(x='Method',y='-log10(Type-I error)') + 
    pn.theme(axis_text_x=pn.element_text(angle=90)) + 
    pn.ggtitle('Total difference to 2.5%/97.5% CDF value expected'))
gg_err_tnorm_utests.save(os.path.join(dir_figures,'err_tnorm_utests.png'),width=5.5,height=3.5)


##########################################
# ---- (ii) SOLVER COVERAGE/RUNTIME ---- #

# Answer: ....



# err2 = 
# res_solver_utests.pivot(index=cn_idx+cn_eval+['mu'],columns='bound',values='value').reset_index('mu').assign(coverage=lambda x: (x['mu']>x['lb']) & (x['mu']<x['ub']) ).groupby(cn_eval)['coverage'].mean()



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

