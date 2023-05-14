"""
Workhorse classes for doing marginal screening (posi_marginal_screen) and Lasso inference (posi_lasso)
"""

# External
import numpy as np
import pandas as pd
from math import isclose
from warnings import warn
from glmnet.linear import ElasticNet
from sklearn.model_selection import KFold
# Internal
from sntn.dists import tnorm, nts
from sntn._split import _split_yx
from sntn.utilities.linear import ols
from sntn.utilities.utils import get_valid_kwargs_cls, cvec, rvec


class _posi_lasso(_split_yx):
    def __init__(self, lam:float, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(y, x, **get_valid_kwargs_cls(_split_yx, **kwargs))
        """
        Carries out selective inference for the Lasso problem

        Parameters
        ==========
        lam:                Regularization penality
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        **kwargs:           Passed into _split_yx(**kwargs), Lasso(**kwargs) and ols(**kwargs)
        """
        # Input checks
        assert lam > 0, 'lam needs to be greater than zero'
        self.lam = lam
        # Run the Lasso
        xtil, self.xmu_screen, self.xstd_screen = self._normalize_x(self.x_screen, return_mu_sd=True)
        lam_max = np.max( np.abs(xtil.T.dot(self.y_screen)) / len(xtil) )
        kwargs_lasso = {'alpha':1, 'n_splits':0, 'standardize':False, 'lambda_path':np.array([lam_max,lam])}
        kwargs_lasso = {**kwargs_lasso, **get_valid_kwargs_cls(ElasticNet, **kwargs)}
        self.mdl_lasso = ElasticNet(**kwargs_lasso)
        self.mdl_lasso.fit(X=xtil, y=self.y_screen)
        bhat_lasso = self.mdl_lasso.coef_path_[:,1]
        self.cidx_screen = np.where(bhat_lasso != 0)[0]
        self.n_cidx_screened = len(self.cidx_screen)
        if self.n_cidx_screened >= 1:
            # Fit an OLS model on the screened and split portion of the data
            ols_kwargs = get_valid_kwargs_cls(ols, **kwargs)
            self.ols_screen = ols(self.y_screen, self.x_screen[:,self.cidx_screen], **ols_kwargs)
            if self.frac_split > 0:
                n_split = self.x_split.shape[0]
                if n_split > self.n_cidx_screened + 1:
                    self.ols_split = ols(self.y_split, self.x_split[:,self.cidx_screen], **ols_kwargs)
                else:
                    warn(f'A total of {self.n_cidx_screened} columns were screened for testing, but we only have {n_split} observations (classical inference is not possible)')
        else:
            warn(f'No variables were screened')
        

    @staticmethod
    def _normalize_x(x:np.ndarray, return_mu_sd:bool=False):
        """Internal method to normalize x and store scaling terms, if necessary"""
        mu, sd = x.mean(axis=0), x.std(axis=0)
        xtil = (x-mu)/sd
        if return_mu_sd:
            return xtil, mu, sd
        else:
            return xtil


    def estimate_sigma2(self, num_folds:int=5) -> None:
        """
        Estimates the variance of residual of the Lasso model. For the screened portion of the data, this amounts to doing k-fold CV

        Parameters
        ==========
        num_folds:                  How many folds of CV to use (note that seed will be inherited from construction)

        Attributes
        ==========
        sig2hat:                    Weighted average of the (unbiased) split estimate + CV-based screening
        se_bhat_screen:             Standard error of the beta coefficients using sig2hat for the screened estimate
        se_bhat_split:              Standard error of the beta coefficients using sig2hat from the split estimate (if frac_split>0)
        """
        # Estimate the variance using K-Fold CV
        folder = KFold(n_splits=num_folds, shuffle=True, random_state=self.seed)
        y_hat = np.zeros(self.y_screen.shape)
        for ridx, tidx in folder.split(self.x_screen):
            x_train, x_test = self.x_screen[ridx], self.x_screen[tidx]
            y_train, y_test = self.y_screen[ridx], self.y_screen[tidx]
            tmp_model = self.__class__(lam=self.lam, y=y_train, x=x_train, frac_split=0.0)
            y_hat[tidx] = tmp_model.ols_screen.predict(x_test[:,tmp_model.cidx_screen])
        sig2hat_screen = np.mean((self.y_screen - y_hat)**2)
        
        # If carving exists, use it
        sig2hat_split = 0
        if self.frac_split > 0:
            sig2hat_split = self.ols_split.sig2hat
        
        # Use data-weighted final value
        self.sig2hat = (1-self.frac_split)*sig2hat_screen + self.frac_split*sig2hat_split

    def get_A(self) -> np.ndarray:
        """
        Calculates the A from Ay <= b for the Lasso procedure
        
        Returns
        =======
        mat_A:          A (..., n) matrix
        """
        # ...
        breakpoint()
        mat_A = np.zeros(10)
        return mat_A
        
    def get_b(self) -> np.ndarray:
        """
        Calculates the b from Ay <= b for the Lasso procedure
        
        Returns
        =======
        vec_b:          A (..., ) array
        """
        # ...
        vec_b = np.zeros(10)
        return vec_b

    def inference_on_screened(self, sigma2:float or int) -> tuple:
        """
        Runs post selection inference for the screened variables

        Returns
        =======
        A tuple containing (alph_den, v_neg, v_pos) which is equivalent to the tnorm(; sigma2, a, b) terms
        """
        # See Lee (2016) for derivations of key terms
        mat_A = self.get_A()
        vec_b = self.get_b()
        # x_s = self.x_screen[:,self.cidx_screen]

