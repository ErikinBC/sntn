"""
Show CIs for diabetes dataset

python3 -m examples.5_diabetes
"""

# External
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
# Internal
from sntn.posi import lasso
from parameters import dir_figures, seed
from sntn.utilities.utils import get_CI, pn_labeller

# ~ Parameters ~ #
lam_frac = 0.25  # What percent of lambda max to use for inference?
alpha = 0.1  # Type-I error rate

##############################
# --- (1) DATA LOAD/PREP --- #

# Load the diabetes data
diabetes = load_diabetes()
y = diabetes.target.copy()
x = diabetes.data.copy()
di_x_cn = dict(zip(range(x.shape[1]), diabetes.feature_names))

# Normalize data
x_til = (x-x.mean(0)) / x.std(0)
y_til = (y - y.mean()) / y.std()

# Determine lambda_max
lam_max = np.max(np.abs(x_til.T.dot(y_til) / len(x_til)))
lam_max10 = lam_frac * lam_max

##########################
# --- (1) PoSI LASSO --- #

# Fit Lasso on 100% of the data
posi_lasso = lasso(lam_max10, y_til, x_til, frac_split=0.0)
# Estimate sigma since we don't know it
posi_lasso.estimate_sigma2()
# Run inference 
posi_lasso.run_inference(alpha=alpha, null_beta=0, run_split=False, run_carve=False)
res_lasso = posi_lasso.res_screen.assign(frac=0, sigma2lasso=True, mdl='lasso')
res_lasso.round(3)


############################
# --- (2) Carved LASSO --- #

frac_split_seq = [0.15, 0.20, 0.25]
holder_carve = []
for frac_split in frac_split_seq:
    tmp_posi = lasso(lam_max10, y_til, x_til, frac_split=frac_split, seed=seed)
    # Try inference with custon sigma2
    tmp_posi.estimate_sigma2()
    tmp_posi.run_inference(alpha=alpha, null_beta=0)
    tmp_res1 = tmp_posi.res_carve.assign(frac=frac_split, sigma2lasso=False)
    # Repeat with lasso-only sigma2
    tmp_posi.run_inference(alpha=alpha, null_beta=0, sigma2=posi_lasso.sig2hat)
    tmp_res2 = tmp_posi.res_carve.assign(frac=frac_split, sigma2lasso=True)
    # Save
    holder_carve.append(tmp_res1)
    holder_carve.append(tmp_res2)
res_carve = pd.concat(holder_carve).reset_index(drop=True)
res_carve = res_carve.assign(mdl='carve')


#######################
# --- (3) ANALYZE --- #

# Plotting libraries
import plotnine as pn
from mizani.formatters import percent_format

# Combine results
res_ribo = pd.concat(objs=[res_lasso, res_lasso.assign(sigma2lasso=False), res_carve]).reset_index(drop=True)
res_ribo['cidx'] = res_ribo['cidx'].map(di_x_cn)
# Select columns where we have two values
tmp_cidx = res_ribo.groupby('cidx')['mdl'].nunique()
tmp_cidx = tmp_cidx[tmp_cidx > 1].index.to_list()
# res_ribo = res_ribo[res_ribo['cidx'].isin(tmp_cidx)].reset_index(drop=True)
cidx_ord = res_ribo.groupby('cidx')['bhat'].mean().sort_values().index
res_ribo['cidx'] = pd.Categorical(res_ribo['cidx'], cidx_ord)

# Use independent sigma
res_ribo_sigma2j = res_ribo.query('~sigma2lasso').drop(columns='sigma2lasso')
res_ribo_sigma2j.pivot(index='cidx',columns='frac',values='pval').round(2)

# Plot comparable CIs
lbls = [f'Carving (split={s*100:.0f}%)' for s in frac_split_seq] + ['PoSI']
lvls = [str(s) for s in frac_split_seq] + ['0.0']
di_split = dict(zip(lvls, lbls))
posd = pn.position_dodge(0.5)
gg_ribo_CI = (pn.ggplot(res_ribo_sigma2j, pn.aes(x='cidx',y='bhat',color='frac.astype(str)')) + 
    pn.theme_bw() + 
    # pn.geom_point(position=posd) + 
    pn.geom_linerange(pn.aes(ymin='lb',ymax='ub'),position=posd) + 
    pn.geom_hline(yintercept=0,linetype='--') + 
    pn.scale_color_discrete(name='Inference',labels=lambda x: [di_split.get(z) for z in x]) + 
    pn.labs(y='Confidence interval',x='Feature') + 
    pn.theme(axis_text_x=pn.element_text(angle=90)))
gg_ribo_CI.save(os.path.join(dir_figures, 'diabetes.png'), width=4.5, height=3.4)


