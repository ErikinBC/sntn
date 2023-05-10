"""
Workhorse classes for doing marginal screening (posi_marginal_screen) and Lasso inference (posi_lasso)
"""

# External
import numpy as np
import pandas as pd
from math import isclose
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# Internal
from sntn.dists import tnorm, nts
from sntn._carve import _carve_yx
from sntn.utilities.linear import ols
from sntn.utilities.utils import get_valid_kwargs_cls, cvec, rvec


class _posi_marginal_screen(_carve_yx):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(y, x, **get_valid_kwargs_cls(_carve_yx, **kwargs))
        """
        Carries out selective inference for a marginal screening proceedure (i.e. top-k correlated features)

        Parameters
        ==========
        k:                  Number of top covariates to pick
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        **kwargs:           Passed into _carve_yx(**kwargs) and ols(**kwargs)


        Methods
        =======
        estimate_sigma2:    Can use "carved" data to estimate the variance of the model (otherwise will need to be specified)
        run_inference:      Carries out post-selection inference on data
        """
        # Input checks
        assert isinstance(k, int) and (k > 0), 'k must be an int that is greater than zero'
        rank = min(self.p, self.x_screen.shape[0])
        assert k < rank, f'For screening to be relevant, k must be less than the rank {rank} (min(n,p))'

        self.k = k

        # Use the "screening" portion of the data to find the top-K coefficients
        z = self.x_screen.T.dot(self.y_screen)
        self.s = np.sign(z)
        self.order = pd.Series(np.abs(z)).sort_values().index.to_list()
        self.cidx_screen = self.order[-self.k:]
        
        # Fit an OLS model on the screened and carved portion of the data
        ols_kwargs = get_valid_kwargs_cls(ols, **kwargs)
        self.ols_screen = ols(self.y_screen, self.x_screen[:,self.cidx_screen], **ols_kwargs)
        if self.frac_carve > 0:
            self.ols_carve = ols(self.y_carve, self.x_carve[:,self.cidx_screen], **ols_kwargs)


    def estimate_sigma2(self, num_folds:int=5) -> None:
        """
        Estimates the variance of residual of the marginally screened model. For the screened portion of the data, this amounts to doing k-fold CV

        Parameters
        ==========
        num_folds:                  How many folds of CV to use (note that seed will be inherited from construction)

        Attributes
        ==========
        sig2hat:                    Weighted average of the (unbiased) carved estimate + CV-based screening
        se_bhat_screen:             Standard error of the beta coefficients using sig2hat for the screened estimate
        se_bhat_carve:              Standard error of the beta coefficients using sig2hat for the carved estimate (if frac_carve>0)
        """
        # Estimate the variance using K-Fold CV
        folder = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        y_hat = np.zeros(self.y_screen.shape)
        for ridx, tidx in folder.split(self.x_screen):
            x_train, x_test = self.x_screen[ridx], self.x_screen[tidx]
            y_train, y_test = self.y_screen[ridx], self.y_screen[tidx]
            tmp_model = self.__class__(k=self.k, y=y_train, x=x_train, frac_carve=0.0)
            y_hat[tidx] = tmp_model.ols_screen.predict(x_test[:,tmp_model.cidx_screen])
        sig2hat_screen = np.mean((self.y_screen - y_hat)**2)
        
        # If carving exists, use it
        sig2hat_carve = 0
        if self.frac_carve > 0:
            sig2hat_carve = self.ols_carve.sig2hat
        
        # Use data-weighted final value
        self.sig2hat = (1-self.frac_carve)*sig2hat_screen + self.frac_carve*sig2hat_carve
        
        # Calculate the se(beta) using the variance
        self.se_bhat_screen = np.sqrt(self.sig2hat * self.ols_screen.igram.diagonal())
        self.se_bhat_carve = np.zeros(self.se_bhat_screen.shape)
        if self.frac_carve > 0:
            self.se_bhat_carve = np.sqrt(self.sig2hat * self.ols_carve.igram.diagonal())
        

    def get_A(self) -> None:
        """
        Calculates the A from Ay <= b for the marginal screening proceedure
        
        Methods
        =======
        mat_A:          A (2*k*(p-k), n) matrix
        """
        # Create a (p,p) Identity matrix and remove the "selected" column rows
        partial = np.identity(self.p)[self.order[:-self.k]]
        partial = np.vstack([partial, -partial])  # (2*(p-k),p) matrix
        n_partial = partial.shape[0]
        # For each screened variable, another partial matrix will be added
        self.mat_A = np.zeros([partial.shape[0]*self.k, self.x_screen.shape[0]])
        for i, cidx in enumerate(self.cidx_screen[::-1]):
            partial_cp = partial.copy()
            partial_cp[:,cidx] = -self.s[cidx]
            start, stop = int(i*n_partial), int((i+1)*n_partial)
            self.mat_A[start:stop] = np.dot(partial_cp, self.x_screen.T)
        


    def run_inference(self, alpha:float, null_beta:float or np.ndarray=0, sigma2:float or None=None) -> pd.DataFrame:
        """
        Carries out inferences...
        
        Parameters
        ==========
        alpha:                  The type-I error rate
        null_beta:              The null hypothesis (default=0)
        sigma2:                 If estimate_sigma2() has not been run, user must specify the value

        Attributes
        ==========
        """
        # -- (i) Input checks -- #
        assert isinstance(alpha, float), 'alpha must be a float'
        assert (alpha > 0) & (alpha < 1), 'alpha must strictly be between (0,1)'
        assert isinstance(null_beta, float) or isinstance(null_beta, int) or isinstance(null_beta, np.ndarray), 'null_beta must be a float/int/array'
        # Broadcast null_beta to match k
        if isinstance(null_beta, float) or isinstance(null_beta, int):
            null_beta = np.repeat(null_beta, self.k)
        if not isinstance(null_beta, np.ndarray):
            null_beta = np.asarray(null_beta)
        assert null_beta.shape[0] == self.k, f'Length of null_beta must '
        if sigma2 is None:
            assert hasattr(self, 'sig2hat'), 'if sigma2 is not specified, run estimate_sigma2 first'
            sigma2 = getattr(self, 'sig2hat')
        assert isinstance(sigma2, float), 'sigma2 must be a float'

        
        # -- (ii) Calculate truncated normal for screened distributioun -- #
        # See Lee (2014) for derivations of key terms
        x_s = self.x_screen[:,self.cidx_screen]
        assert all([isclose(xmu,0) for xmu in x_s.mean(0)]), 'Expected x_s to be de-meaned'
        # eta: The direction column span of x to be tested (k,n)
        idx_int = int(self.ols_screen.has_int)
        bhat_S = self.ols_screen.bhat[idx_int:]
        eta = np.dot(self.ols_screen.igram[idx_int:,idx_int:], x_s.T)
        # see Eq. 6 (note that sigma2 should cancel out)
        alph_num = sigma2 * np.dot(self.mat_A, eta.T)
        alph_den = sigma2 * np.sum(eta**2, axis=1)
        # A (q,k) array, where q is the number of constraints for each of the k dimensions being tested
        alph = alph_num / alph_den
        # mat_A is a (q,n) set of constaints on each of the dimensions of y
        # Ay is a (q,1) array, it is independent of the k coordinates
        Ay = np.dot(self.mat_A, cvec(self.y_screen))
        # For each column, find the alph's that are negative and apply Eq. 7, the alph's the are positive and apply Eq. 8
        ratio = -Ay/alph + rvec(bhat_S)  # alph_j*beta_j/alph_j = beta_j
        assert ratio.shape == alph.shape, 'ratio and alph should be the same shape'
        # Indexes are w.r.t to alph's
        v_neg = np.max(np.where(alph < 0, ratio, -np.inf), axis=0)
        v_pos = np.min(np.where(alph > 0, ratio, +np.inf), axis=0)
        assert np.all(v_pos > v_neg), 'expected pos to be larger than neg'
        
        # Create a dataframe with the truncnorm distribution
        self.res_screen = pd.DataFrame({'cidx':self.cidx_screen,'x':bhat_S, 'a':v_neg, 'b':v_pos, 'sig2':alph_den})
        self.dist_screen = tnorm(null_beta, alph_den, v_neg, v_pos)
        # Add on the p-values and conf-int's
        pval = self.dist_screen.cdf(bhat_S)
        pval = 2*np.minimum(pval, 1-pval)
        self.res_screen['pval'] = pval
        ci_lbub = self.dist_screen.conf_int(bhat_S, alpha, a=v_neg, b=v_pos, sigma2=alph_den)
        self.res_screen['ci_lb'] = ci_lbub[:,0]
        self.res_screen['ci_ub'] = ci_lbub[:,1]

        # self.dist_carve = None
        # if self.frac_carve > 0:
        #     self.dist_carve = nts(mu1=,tau21=,mu2=,tau22=,a=,b=,)
        
        

