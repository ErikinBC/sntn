######################
# --- (1) SET UP --- #

# python3 -m examples.data_carving

# Load external packages
import numpy as np
import pandas as pd
from scipy.stats import norm
from glmnet import ElasticNet
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# Load SNTN utilities
from sntn.dists import nts, tnorm
from sntn.posi import lasso, marginal_screen

# Checking numerical stability of solutions
kkt_tol = 5e-3

# Set up null-hypothesis testing
alpha = 0.05
null_beta = 0

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
cn_X = load_breast_cancer().feature_names
n, p = X.shape

# Split data into screening and inference
pct_split = 0.5
seed = 1
X_B, X_A, y_B, y_A = train_test_split(X, y, test_size=pct_split, random_state=seed)
n_A, n_B = len(y_A), len(y_B)
w_A, w_B = n_A/n, n_B/n
# Normalize B (which will go into algorithms)
xmu = X_B.mean(0)
xse = X_B.std(0,ddof=1)
Xtil_B = (X_B - xmu) / xse
# Put A on the same scale as B so the coefficients have the same intepretation
Xtil_A = (X_A - xmu) / xse

# De-mean y so we don't need to fit intercept
ytil_A = y_A - y_A.mean()
ytil_B = y_B - y_B.mean()


#####################
# --- (2) LASSO --- #

# (i) Run the Lasso for some fixed value of lambda
lamb_fix = 1e-2
lamb_path = np.array([lamb_fix*1.01, lamb_fix])  # ElasticNet wants >=2 values
mdl_lasso = ElasticNet(alpha=1, n_splits=0, standardize=False, lambda_path=lamb_path, tol=1e-20)
mdl_lasso.fit(Xtil_B, ytil_B)
bhat_lasso = mdl_lasso.coef_path_[:,1]
M_lasso = bhat_lasso != 0
cn_lasso = cn_X[M_lasso]
X_B_M = Xtil_B[:,M_lasso].copy()
# Check KKT conditions
kkt_lasso = np.abs(Xtil_B.T.dot(ytil_B - Xtil_B.dot(bhat_lasso)) / n_B)
err_kkt = np.max(np.abs(kkt_lasso[M_lasso] - lamb_fix))
assert err_kkt < kkt_tol, 'KKT conditions not met'
assert np.all(lamb_fix > kkt_lasso[~M_lasso]), 'KKT conditions not met'
n_sel = M_lasso.sum()
print(f'Lasso selected {n_sel} of {p} features for lambda={lamb_fix} (KKT-err={err_kkt:0.5f})')

# (ii) Run OLS on the screening portion
igram_B_M = np.linalg.inv(np.dot(X_B_M.T, X_B_M))
eta_B_M = np.dot(X_B_M, igram_B_M)
bhat_B = np.dot(eta_B_M.T, ytil_B)

# (iii) Run inference on the classical held-out portion
X_A_M = Xtil_A[:,M_lasso].copy()
igram_A_M = np.linalg.inv(np.dot(X_A_M.T, X_A_M))
eta_A_M = np.dot(X_A_M, igram_A_M)
bhat_A = np.dot(eta_A_M.T, ytil_A)

# (iv) Use held-out portion to estimate variance of the error
resid_A = ytil_A - np.dot(X_A_M, bhat_A)
sigma2_A = np.sum(resid_A**2) / (len(X_A) - M_lasso.sum())
tau21 = sigma2_A * np.sum(eta_A_M**2,0)

# (v) Calculate the polyhedral constraints
sign_M = np.diag(np.sign(bhat_lasso[M_lasso]).astype(int))
A = -np.dot(sign_M, eta_B_M.T)
# Note, that lambda is multiplied by n_B since the solution below solves for the non-normalized version
b = -n_B * lamb_fix * np.dot(sign_M, np.dot(igram_B_M, sign_M.diagonal()))
# Check for constraint error
assert  np.all(A.dot(ytil_B) <= b), 'Polyhedral constraint not met!'

# (vi) Calculate the truncated normal values (4.8 to 5.6 for Lee's paper, using only A1/A2)
tau22 = sigma2_A * np.sum(eta_B_M**2,0)
D = np.sqrt(tau22 / sigma2_A)
V = eta_B_M / D
R = np.squeeze(np.dstack([np.dot(np.diag(np.ones(n_B)) - np.outer(V[:,j],V[:,j]),ytil_B)  for j in range(n_sel)]))
AV = np.dot(A, V)
AR = A.dot(R)
nu = (b.reshape([n_sel,1]) - AR) / AV
idx_neg = AV < 0
idx_pos = AV > 0
a = np.max(np.where(idx_neg, nu, -np.inf),0) * D
b = np.min(np.where(idx_pos, nu, +np.inf),0) * D

# Signs can be used to determine direction of hypothesis test
is_sign_neg = np.sign(bhat_lasso[M_lasso])==-1

# (i) PoSI with the Lasso
dist_tnorm = tnorm(null_beta, tau22, a, b)
pval_tnorm = dist_tnorm.cdf(bhat_B)
pval_tnorm = np.where(is_sign_neg, pval_tnorm, 1-pval_tnorm)
res_tnorm = pd.DataFrame({'mdl':'Screening','cn':cn_lasso,'bhat':bhat_B, 'pval':pval_tnorm})
res_tnorm = pd.concat(objs=[res_tnorm, pd.DataFrame(dist_tnorm.conf_int(bhat_B, alpha),columns=['lb','ub'])],axis=1)

# (ii) OLS on inference set
dist_norm = norm(loc=null_beta, scale=np.sqrt(tau21))
pval_norm = dist_norm.cdf(bhat_A)
pval_norm = np.where(is_sign_neg, pval_norm, 1-pval_norm)
res_norm = pd.DataFrame({'mdl':'Splitting','cn':cn_lasso, 'bhat':bhat_A, 'pval':pval_norm})
res_norm = pd.concat(objs=[res_norm, pd.DataFrame(np.vstack(norm(bhat_B, np.sqrt(tau21)).interval(1-alpha)).T,columns=['lb','ub'])],axis=1)


# (iii) Data carving on both
bhat_wAB = w_A*bhat_A + w_B*bhat_B
dist_sntn = nts(null_beta, tau21, null_beta, tau22, a, b, w_A, w_B, fix_mu=True)
pval_sntn = dist_sntn.cdf(bhat_wAB)
pval_sntn = np.where(is_sign_neg, pval_sntn, 1-pval_sntn)
res_sntn = pd.DataFrame({'mdl':'Carving','cn':cn_lasso,'bhat':bhat_wAB, 'pval':pval_sntn})
res_sntn = pd.concat(objs=[res_sntn, pd.DataFrame(np.squeeze(dist_sntn.conf_int(bhat_wAB, alpha)),columns=['lb','ub'])],axis=1)

# Combine all
res_lasso = pd.concat(objs=[res_tnorm, res_norm, res_sntn], axis=0).reset_index(drop=True)
res_lasso = res_lasso.assign(is_sig=lambda x: np.where(x['pval'] < alpha, True, False))


# Repeat with the lasso wrapper
wrapper_lasso = lasso(lamb_fix, y, X, frac_split=pct_split, seed=seed)
wrapper_lasso.run_inference(alpha, null_beta ,sigma2_A)
wrapper_lasso.res_carve
wrapper_lasso.res_split
wrapper_lasso.res_screen
breakpoint()



##################################
# --- (3) MARGINAL SCREENING --- #

