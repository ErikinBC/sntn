"""
Make sure PoSI classes work as expected

python3 -m tests.test_posi
python3 -m pytest tests/test_posi.py -s
"""

# External
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm
# from sklearn.linear_model import ElasticNetCV
# Internal
from sntn.posi import marginal_screen
from sntn.utilities.linear import dgp_sparse_yX, ols


def test_posi(k:int=5, nsim:int=1000, n:int=100, p:int=150, s:int=5, b0:int=+1, snr:float=1.0, pval_exact:float=0.05, cover_err_est:float=0.04, tol_exact:float=1e-3, tol_approx:float=1e-3 ) -> None:
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
    tol_exact:          For the "exact" distributions, how close should actual p-value be to theoretical
    tol_approx:         For the "approximate" distributions, how close should actual p-value be to theoretical

    Checks that:
    i) ...
    """
    from scipy.stats import binom

    # --- (i) Generate data and run simulations --- #
    cidx_gt = pd.Series(range(s))  # Which coefficients are non-noise
    holder_sim = []
    for i in range(nsim):
        # Draw data
        if (i+1) % 250 == 0:
            print(f'Simulation {i+1} of {nsim}')
        y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, return_params=True, seed=i)
        screener = marginal_screen(k, y, x, seed=i)
        
        # Extract the coefficients
        beta_screen_i = screener.ols_screen.bhat[1:]
        beta_carve_i = screener.ols_carve.bhat[1:]
        
        # Assume we know the oracle variance var(y-f(xbeta))
        n_screened = cidx_gt.isin(screener.cidx_screen).sum()
        sigma2_posi = 1 + (s - n_screened)*beta1[0]**2
        
        # Use the groundtruth variance
        gt_screen_i = np.sqrt(sigma2_posi * screener.ols_screen.igram.diagonal()[1:])
        gt_carve_i = np.sqrt(sigma2_posi * screener.ols_carve.igram.diagonal()[1:])
        
        # Use estimated variance
        screener.estimate_sigma2()
        se_screen_i = screener.se_bhat_screen[1:]
        se_carve_i = screener.se_bhat_carve[1:]
        
        # Save for later
        sim_i = pd.DataFrame({'beta1':beta1[screener.cidx_screen],'bhat_screen':beta_screen_i, 'bhat_carve':beta_carve_i,'se_screen':se_screen_i, 'se_carve':se_carve_i, 'gt_screen':gt_screen_i, 'gt_carve':gt_carve_i},index=screener.cidx_screen).assign(sim=i)
        holder_sim.append(sim_i)
    
    # --- (ii) Merge data and get p-values --- #
    cn_idx = ['sim','cidx','beta1']
    res_sim = pd.concat(holder_sim).rename_axis('cidx').reset_index().melt(cn_idx,var_name='tmp')
    tmp = res_sim['tmp'].str.split(pat='\\_',n=1,expand=True).rename(columns={0:'param',1:'tt'})
    res_sim = pd.concat(objs=[res_sim.drop(columns='tmp'), tmp], axis=1)
    # Calculate naive z's
    res_sim = res_sim.pivot(index=cn_idx+['tt'],columns='param',values='value')
    res_sim.reset_index(inplace=True)
    # Assign ground-truth coefficient to zero if outside the cidx
    res_sim = res_sim.assign(z_est=lambda x: (x['bhat']-x['beta1'])/x['se'], z_gt=lambda x: (x['bhat']-x['beta1'])/x['gt'])
    res_sim.drop(columns=['se','gt'], inplace=True)
    # Replace beta1 with noise or real
    res_sim = res_sim.rename(columns={'beta1':'noise'}).assign(noise=lambda x: x['noise']==0)
    cn_idvar = ['tt','cidx','noise','sim']
    res_sim = res_sim.melt(cn_idvar,['z_est','z_gt'],var_name='sigma2',value_name='z_null')
    res_sim['sigma2'] = res_sim['sigma2'].str.replace('z_','',regex=False)
    res_sim = res_sim.assign(pval=lambda x: 2*norm.cdf(-x['z_null'].abs()))
        
    # --- (iii) Check p-values (carved) --- #
    p_check = np.arange(0.1, 1, 0.1)
    alpha_seq = [0.05, 0.1, 0.2]
    cn_msr = ['sigma2','noise','z_null','pval']
    cn_gg = ['sigma2','noise']
    data_carved = res_sim.loc[res_sim['tt']=='carve',cn_msr].reset_index(drop=True)
    # Check coverage with p-values
    holder_carved = []
    for alpha in alpha_seq:
        tmp_cover = data_carved.assign(reject=lambda x: x['pval']<alpha).groupby(cn_gg)['reject'].agg({'sum','count'})
        dist_cover = binom(n=tmp_cover['count'], p=alpha)
        pval_cover = dist_cover.cdf(tmp_cover['sum'])
        pval_cover = 2*np.minimum(pval_cover, 1-pval_cover)
        tmp_res = pd.DataFrame({'pval':pval_cover, 'emp':tmp_cover['sum']/tmp_cover['count'], 'alpha':alpha}, index=tmp_cover.index)
        holder_carved.append(tmp_res)
    res_carved = pd.concat(holder_carved).reset_index()
    pval_gt_carved = res_carved.loc[res_carved['sigma2']=='gt','pval'].min()
    assert pval_gt_carved > pval_exact, f'The smallest p-value from the coveraged test for carved data {pval_gt_carved} is smaller than tolerance: {pval_exact}'
    cover_est_carved = res_carved.loc[res_carved['sigma2']=='est',['alpha','emp']].diff(axis=1).iloc[:,1].abs().max()
    assert cover_est_carved < cover_err_est, f'The largest different in coverage for carved data {cover_est_carved} is larger than tolerance: {cover_err_est}'

    
    # --- (iv) Check p-values and coverage (screened) --- #


    # --- (v) Check p-values and coverage (carved + screened) --- #

    
    
    