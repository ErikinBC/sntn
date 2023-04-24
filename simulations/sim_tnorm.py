"""
Simulations (analyses) for the tnorm class:

i) Which solvers should be considered for further testing?
ii) Which solvers get the right coverage and which is the fastest?
iii) Does the built in "fit" method work?

python3 -m simulations.sim_tnorm
"""

# External
import os
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
# Internal
from sntn.dists import tnorm
from tests.test_dists_tnorm import gen_params
from parameters import dir_figures, dir_simulations, seed
from sntn.utilities.utils import mean_total_error, cat_by_order, pn_labeller
# For multiindex slicing
idx = pd.IndexSlice

# Hard-coded terms
alpha = 0.05
di_bound = {'lb':'Lower-bound', 'ub':'Upper-bound'}
cn_eval = ['approach','method']  # What we are assessing
cn_idx = ['n','ndraw','idx']  # For pivoting


################################
# ---- (i) SCREEN SOLVERS ---- #

# Answer: Loading the unittests, we find that 9/19 have reasonable type-I error control: minimize-Powell, minimize_scalar-Golden, minimize_scalar-Brent, root_scalar-toms748, root_scalar-secant, root_scalar-brenth, root_scalar-brentq, root_scalar-ridder, root_scalar-bisect

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
err_val = err_val.assign(nlog10=lambda x: -np.log10(x['pval_err']))

gg_err_tnorm_utests = (pn.ggplot(err_val, pn.aes(x='method',y='nlog10',color='approach')) + 
    pn.theme_bw() + pn.geom_point(size=2) + 
    pn.scale_color_discrete(name='scipy approach') + 
    pn.labs(x='Method',y='-log10(Type-I error)') + 
    pn.theme(axis_text_x=pn.element_text(angle=90)) + 
    pn.ggtitle('Total difference to 2.5%/97.5% CDF value expected'))
gg_err_tnorm_utests.save(os.path.join(dir_figures,'err_tnorm_utests.png'),width=5.5,height=3.5)

# Determine the "subset" to try
methods_test = err_val.query('nlog10 > 4').copy()
n_methods = len(methods_test)
print(f'A total of {n_methods} out of {len(err_val)} approach/methods will be further tested')
print(', '.join(methods_test.apply(lambda x: f"{x['approach']}-{x['method']}", 1).to_list()))


##########################################
# ---- (ii) SOLVER COVERAGE/RUNTIME ---- #

# Answer: default should be root_scalar:secant, with minimize_scalar:Golden as a backup for slower but slightly more accurate

# Simulation parameters
n_draw = np.arange(5, 100+1, 5)
n_sim = len(n_draw)
stime = time()
holder_methods = []
holder_params = []
for j, n in enumerate(n_draw):
    # Recycle parameters across draws 
    mu, sigma2, a, b = gen_params((n,), seed)
    dist = tnorm(mu, sigma2, a, b)
    x = np.squeeze(dist.rvs(1))
    # Store for evaluation
    dat_params = pd.DataFrame({'x':x, 'mu':mu, 'sigma2':sigma2, 'a':a, 'b':b, 'n':n})
    holder_params.append(dat_params)
    for i, r in methods_test.iterrows():
        approach, method = r['approach'], r['method']
        di_scipy = {'method':method}
        # Fit and draw data
        ci_stime = time()
        if method == 'newton':
            res = dist.conf_int(x=x, alpha=alpha, approach=approach, di_scipy=di_scipy, approx=True, sigma2=sigma2, a=a, b=b, a_min=1e-5, a_max=np.inf)
        else:
            res = dist.conf_int(x=x, alpha=alpha, approach=approach, di_scipy=di_scipy, approx=True, sigma2=sigma2, a=a, b=b)
        ci_time = time() - ci_stime
        res = pd.DataFrame(res,columns=['lb','ub'])
        res = res.assign(approach=approach, method=method, dtime=ci_time, n=n)
        holder_methods.append(res)
        # Print ETA        
        dtime = time() - stime
        n_progress = j*n_methods + (i+1)
        n_iter_left = n_sim*n_methods - n_progress
        rate = n_progress / dtime
        eta = n_iter_left / rate
        print(f'Iteration {n_progress}, {n_iter_left} left of ETA = {eta:.0f} seconds ({eta/60:.1f} minutes)')
# Merge results
res_runtime = pd.concat(holder_methods).rename_axis('idx').reset_index()
res_params = pd.concat(holder_params).rename_axis('idx').reset_index()
# Get the associated p-value
res_runtime_wide = res_runtime.pivot(index=['n','idx'],columns=['approach','method'],values=['lb','ub'])
res_runtime_wide.columns.names = pd.Series(res_runtime_wide.columns.names).replace({None:'bound'})
dists_wide = tnorm(res_runtime_wide.values, res_params[['sigma2']], res_params[['a']], res_params[['b']])
val_pval_wide = dists_wide.cdf(res_params[['x']])
val_pval_wide = pd.DataFrame(val_pval_wide, index=res_runtime_wide.index, columns = res_runtime_wide.columns)
val_pval_wide = val_pval_wide.melt(ignore_index=False,value_name='pval_ci').reset_index()
val_pval_wide = val_pval_wide.assign(apval_err=lambda x:  np.abs(x['pval_ci']-(np.where(x['bound']=='lb',1-alpha/2,alpha/2))))
# Merge back with runtime
val_pval_aerr = val_pval_wide.groupby(['n','idx','approach','method'])['apval_err'].sum().reset_index()
res_runtime = res_runtime.merge(val_pval_aerr)
# Save for convenience
path_runtime = os.path.join(dir_simulations,'res_runtime.csv')
res_runtime.to_csv(path_runtime, index=False)
if 'res_runtime' not in dir():
    res_runtime = pd.read_csv(path_runtime)

# Calculate the overall performance (runtime for a given n and cumulative error)
di_msr = {'nlog_err':'-log10(Type-I error)', 'dtime':'Time (seconds)'}
res_runtime_msr = res_runtime.groupby(cn_eval+['n']).agg({'apval_err':'sum', 'dtime':'mean'})
# Get cumulative value
res_runtime_msr = res_runtime_msr.groupby(cn_eval).cumsum()
res_runtime_msr['nlog_err'] = -np.log10(res_runtime_msr['apval_err'])
res_runtime_msr = res_runtime_msr.drop(columns='apval_err').melt(ignore_index=False,var_name='msr').reset_index()
dat_txt = res_runtime_msr.query('n == n.unique()[-3]')

# Plot the cumulative runtime/error
gg_tnorm_time = (pn.ggplot(res_runtime_msr,pn.aes(x='n',y='value',color='approach',group='method')) + 
    pn.theme_bw() + pn.geom_line() + 
    pn.labs(y='Cumulative value',x='# of solutions') + 
    pn.scale_color_discrete(name='scipy approach') + 
    pn.geom_text(pn.aes(label='method'),data=dat_txt,size=10) + 
    pn.facet_wrap('~msr',labeller=pn_labeller(msr=lambda x: di_msr.get(x,x)),scales='free_y') + 
    pn.theme(subplots_adjust={'wspace': 0.25}))
gg_tnorm_time.save(os.path.join(dir_figures, 'tnorm_methods_perf.png'),width=8,height=3.5)

# Plot the trade-off
res_runtime_tradeoff = res_runtime_msr.query('n == n.max()')
res_runtime_tradeoff = res_runtime_tradeoff.pivot_table(index=cn_eval,columns='msr',values='value').reset_index()

gg_tnorm_tradeoff = (pn.ggplot(res_runtime_tradeoff, pn.aes(x='nlog_err',y='dtime',color='approach')) + 
    pn.theme_bw() + pn.geom_point(size=2) + 
    pn.scale_color_discrete(name='scipy approach') + 
    pn.ggtitle('Trade-off in (cumulative) performance/error') + 
    pn.geom_text(pn.aes(label='method'),size=10, adjust_text={
    'expand_points': (2,2)}) + 
    pn.labs(x='-log10(Type-I error)',y='Runtime (seconds)'))
gg_tnorm_tradeoff.save(os.path.join(dir_figures, 'tnorm_tradeoff.png'),width=5.5,height=3.5)


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
    pn.geom_vline(xintercept=[a,b],color='blue',linetype='--') + 
    pn.scale_color_discrete(name='Sample size') + 
    pn.scale_shape_discrete(name='Standard dev') + 
    pn.scale_x_continuous(breaks=mu_seq) + 
    pn.facet_grid('sigma~fix_sigma',labeller=pn.label_both))
gg_tnorm_fit.save(os.path.join(dir_figures, 'tnorm_fit.png'),width=7, height=6)

