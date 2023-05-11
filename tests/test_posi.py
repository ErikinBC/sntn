"""
Make sure PoSI classes work as expected

python3 -m tests.test_posi
python3 -m pytest tests/test_posi.py -s
"""

# External
import pytest
import numpy as np
import pandas as pd
from time import time
from scipy.stats import norm, binom
# from sklearn.linear_model import ElasticNetCV
# Internal
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX, ols


def test_posi(k:int=5, nsim:int=1000, n:int=100, p:int=150, s:int=2, b0:int=+1, snr:float=2.0, type1:float=0.1, pval_gt_tol:float=0.01, cover_gt_tol:float=0.01) -> None:
    """
    Parameters
    ==========
    k:                  Number of top covariates to pick
    nsim:               Number of simulations to run
    n:                  Number of observations
    p:                  Number of covariates
    s:                  Number of covariates with actual signal
    b0:                 Intercept of the response vector
    snr:                The signal to noise ratio = sum_{j=1}^{s} b_j^2
    type1:              Type-I error for CIs to check
    pval_exact:         When looking at the distribution of the null when all params are known, what is the maximum p-value we will allow before rejection?
    cover_err_est:      When there are nuissance parameters, what is the coverage buffer we will permit?
    
    Checks that:
    i) Rejection rate coverage aligns with expected binomial proportion (see pval_gt_tol)
    ii) The coverage of the true parameter aligns with expected binomial proportion (see pval_gt_tol)
    """
    

    # --- (i) Generate data and run simulations --- #
    cn_screen_keep = ['pval','lb','ub']
    cidx_gt = pd.Series(range(s))  # Which coefficients are non-noise
    holder_sim = []
    stime = time()
    for i in range(nsim):
        # Draw data
        if (i+1) % 25 == 0:
            dtime = time() - stime
            rate = (i+1) / dtime
            seta = (nsim-i-1)/rate
            print(f'Simulation {i+1} of {nsim} (ETA={seta:0.0f} seconds)')
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, return_params=True, seed=i)
        
        # (i) Run the screen (do not normalize x since this effectively regularizes bhat coefficients slightly)
        screener = marginal_screen(k, y, x, seed=i, normalize=False, alpha=type1)
        screener.get_A()  # Get the PoSI constraints
        
        # (ii) Calculate the oracle variance var(y-f(xbeta))
        idx_noise = ~np.isin(screener.cidx_screen, cidx_gt)
        n_screened = np.sum(~idx_noise)
        beta_null = beta1[screener.cidx_screen]
        sigma2_posi = 1 + (s - n_screened)*beta1[0]**2
        
        # (iii) Perform inference when variance is "known"
        screener.run_inference(alpha=type1, null_beta=beta_null, sigma2=sigma2_posi)
        res_gt_i = pd.concat(objs=[screener.res_split.assign(mdl='split', sigma2='gt'),
                                   screener.res_screen.assign(mdl='screen', sigma2='gt')])

        # (iv) Perform inference when variance needs to be estimated
        screener.estimate_sigma2()
        screener.run_inference(alpha=type1, null_beta=beta_null, sigma2=screener.sig2hat)
        res_est_i = pd.concat(objs=[screener.res_split.assign(mdl='split', sigma2='est'),
                                   screener.res_screen.assign(mdl='screen', sigma2='est')])
        # Join and save
        res_i = pd.concat(objs=[res_gt_i, res_est_i]).assign(sim=i+1)
        holder_sim.append(res_i)
    
    # --- (ii) Merge data --- #
    res_sim = pd.concat(holder_sim).reset_index(drop=True)
    res_sim['noise'] = ~res_sim['cidx'].isin(cidx_gt)
       
    # --- (iii) Check p-value rejection --- #
    alpha_seq = [0.05, 0.1, 0.2]
    cn_gg = ['mdl','noise']
    dat_pval_exact = res_sim.loc[res_sim['sigma2']=='gt',cn_gg + ['pval']].reset_index(drop=True)
    # Check coverage with p-values
    holder_pval = []
    for alpha in alpha_seq:
        dat_pval_exact['reject'] = dat_pval_exact['pval'] < alpha
        reject_rate = dat_pval_exact.groupby(cn_gg)['reject'].agg({'sum','count'})
        reject_rate = reject_rate.assign(emp=lambda x: x['sum']/x['count'])
        dist_reject = binom(n=reject_rate['count'], p=alpha)
        pval_reject = dist_reject.cdf(reject_rate['sum'])
        pval_reject = 2*np.minimum(pval_reject, 1-pval_reject)
        reject_rate['pval'] = pval_reject
        reject_rate['alpha'] = alpha
        reject_rate.drop(columns=['count','sum'], inplace=True)
        holder_pval.append(reject_rate)
    res_pval = pd.concat(holder_pval).reset_index()
    pval_gt_min = res_pval['pval'].min()
    assert pval_gt_min > pval_gt_tol, f'The smallest p-value using ground truth variance {pval_gt_min} is smaller than tolerance: {pval_gt_tol}'
    
    # --- (iv) Check confidence interval converage --- #
    dat_ci_exact = res_sim.loc[res_sim['sigma2']=='gt',cn_gg + ['lb','ub']].reset_index(drop=True)
    dat_ci_exact['beta1'] = np.where(dat_ci_exact['noise'], 0, beta1[0])
    dat_ci_exact = dat_ci_exact.assign(cover=lambda x: (x['lb'] <= x['beta1']) & (x['ub'] >= x['beta1']))
    cover_rate = dat_ci_exact.groupby(['mdl','noise'])['cover'].agg({'sum','count'})
    cover_rate = cover_rate.assign(emp=lambda x: x['sum']/x['count'])
    dist_cover = binom(n=cover_rate['count'], p=1-type1)
    pval_cover = dist_cover.cdf(cover_rate['sum'])
    pval_cover = 2*np.minimum(pval_cover, 1-pval_cover)
    cover_rate['pval'] = pval_cover
    cover_gt_min = cover_rate['pval'].min()
    assert cover_gt_min > pval_gt_tol, f'The smallest p-value using ground truth variance {cover_gt_min} is smaller than tolerance: {pval_gt_tol}'
    
    
