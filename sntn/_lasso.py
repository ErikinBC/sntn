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
from sntn._cdf_bvn._utils import Phi
from sntn.utilities.linear import ols
from sntn.utilities.utils import get_valid_kwargs_cls, cvec, rvec


class _posi_lasso(_split_yx):
    def __init__(self, lam:float, y:np.ndarray, x:np.ndarray, kkt_tol:float=5e-4, **kwargs):
        super().__init__(y, x, **get_valid_kwargs_cls(_split_yx, **kwargs))
        """
        Carries out selective inference for the Lasso problem

        Parameters
        ==========
        lam:                Regularization penality
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        kkt_tol:            How much floating point error to allow when checking the KKT conditions
        **kwargs:           Passed into _split_yx(**kwargs), Lasso(**kwargs) and ols(**kwargs)
        """
        # Input checks
        assert lam > 0, 'lam needs to be greater than zero'
        self.lam = lam
        self.kkt_tol = kkt_tol
        # Run the Lasso
        xtil, self.xmu_screen, self.xstd_screen = self._normalize_x(self.x_screen, return_mu_sd=True)
        lam_max = np.max( np.abs(xtil.T.dot(self.y_screen)) / len(xtil) )
        kwargs_lasso = {'alpha':1, 'n_splits':0, 'standardize':False, 'tol':1e-20, 'lambda_path':np.array([lam_max,lam])}
        kwargs_lasso = {**kwargs_lasso, **get_valid_kwargs_cls(ElasticNet, **kwargs)}
        self.mdl_lasso = ElasticNet(**kwargs_lasso)
        self.mdl_lasso.fit(X=xtil, y=self.y_screen)
        bhat_lasso = self.mdl_lasso.coef_path_[:,1]
        self.cidx_screen = np.where(bhat_lasso != 0)[0]
        self.k = len(self.cidx_screen)
        self.bhat_M = bhat_lasso[self.cidx_screen]
        self.int_M = self.mdl_lasso.intercept_path_[1]
        self.n_cidx_screened = len(self.cidx_screen)
        if self.n_cidx_screened >= 1:
            # Fit an OLS model on the screened and split portion of the data
            ols_kwargs = get_valid_kwargs_cls(ols, **kwargs)
            self.ols_screen = ols(self.y_screen, self.x_screen[:,self.cidx_screen], **ols_kwargs)
            if self.frac_split > 0:
                n_split = self.x_split.shape[0]
                if n_split >= self.n_cidx_screened + 1:
                    self.ols_split = ols(self.y_split, self.x_split[:,self.cidx_screen], **ols_kwargs)
                else:
                    warn(f'A total of {self.n_cidx_screened} columns were screened for testing, but we only have {n_split} observations (classical inference is not possible)')
        else:
            warn(f'No variables were screened')
        

    @staticmethod
    def _normalize_x(x:np.ndarray, return_mu_sd:bool=False):
        """Internal method to normalize x and store scaling terms, if necessary"""
        mu, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
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


    def _get_Ab(self, partial:bool=True) -> np.ndarray:
        """
        Calculates the A/b from Ay <= b for the Lasso procedure
        
        Parameters
        ==========
        partial:            Whether we want to test the |M| variables independently (and only use hte active constraints)

        Returns
        =======
        A:          A (q, n) matrix
        b:          A (q,) array

        where q = |M| + 2*(p-|M|)
        """
        # Pre-define terms
        x_M = self.x_screen[:,self.cidx_screen]
        
        # (i) Check KKT conditions
        eps_screen = self.y_screen - (self.int_M + x_M.dot(self.bhat_M))
        assert np.abs(np.mean(eps_screen)) < self.kkt_tol
        kkt_subgrab = self.x_screen.T.dot(eps_screen) / self.x_screen.shape[0]
        err_M = np.max(np.abs(kkt_subgrab[self.cidx_screen]) - self.lam)
        assert err_M < self.kkt_tol, f'Expected selected covariates to have inner product values of {self.lam}, found error of {err_M}'
        err_nM = np.min(self.lam - np.abs(np.delete(kkt_subgrab,self.cidx_screen)))
        assert err_nM > 0, f'Expected non-selected variables to have at least {self.kkt_tol} values larger than zero, found {err_nM} instead'

        # (ii) Calculate A1/b1
        n_yscreen = self.y_screen.shape[0]
        if partial:
            sM = np.sign(self.bhat_M)
            x_Mplus = np.dot(self.ols_screen.igram, x_M.T)
            A1 = -np.dot(np.diag(sM), x_Mplus)
            b1 = cvec(-self.lam * np.dot(np.diag(sM), np.dot(self.ols_screen.igram, sM)))
            b1 *= n_yscreen
            assert np.all(A1.dot(self.y_screen) <= b1.flatten()), f'Polyhedral constraint not met for A1 y <= b'
            assert A1.shape[1] == n_yscreen, 'A1 does not align with y'
            return A1, b1

        # (iii) Calculate A0/b0
        cidx_nM = np.delete(np.arange(self.x_screen.shape[1]), self.cidx_screen)
        x_nM = self.x_screen[:,cidx_nM]
        # Identity matrix less projection matrix
        iP_M = np.dot(np.dot(x_M, self.ols_screen.igram), x_M.T)
        iP_M = np.diag(np.ones(iP_M.shape[0])) - iP_M
        A0 = np.dot(x_nM.T, iP_M)
        A0 = (1/self.lam) * np.vstack((A0,-A0))
        b0 = np.dot(np.dot(x_nM.T, x_Mplus.T), sM)
        b0 = cvec(np.hstack((1 - b0, 1 + b0)))
        b0 *= n_yscreen
        assert np.all(A0.dot(self.y_screen) <= b0.flatten()), f'Polyhedral constraint not met for A1 y <= b'
        assert A0.shape[1] == n_yscreen, 'A0 does not align with y'

        # (iv) Combine matrices and return    
        A = np.vstack((A1, A0))
        b = np.vstack((b1, b0))
        return A, b


    def _inference_on_screened(self, sigma2:float | int, partial:bool=True) -> tuple:
        """
        Runs post selection inference for the screened variables

        Parameters
        ==========
        sigma2:             Variance of the error term
        partial:            Whether we want to test the |M| variables independently (and only use hte active constraints)

        Returns
        =======
        A tuple containing (eta2var, v_neg, v_pos) which is equivalent to the tnorm(; sigma2, a, b) terms
        """
        # (i) See Lee (2016) for derivations of key terms
        n, p = self.x_screen.shape
        active = np.isin(np.arange(p), self.cidx_screen)
        n_active = active.sum()
        x_M = self.x_screen[:, active]
        y_M = self.y_screen - self.y_screen.mean()
        I_n = np.diag(np.ones(n))
        eta_T = np.dot(self.ols_screen.igram, x_M.T)
        sign_vars = np.sign(self.bhat_M)
        # Note, b has been adjusted b/c we are solving ||y-Xbeta||_2^2 - lam*||beta||_1 unnormalized by n
        A, b = self._get_Ab(partial)
        
        targets, den = np.zeros(n_active), np.zeros(n_active)
        V_low, V_high = np.zeros(n_active), np.zeros(n_active)
        for i in range(n_active):
            # Calculate
            eta_i = eta_T[[i]].T
            den_i = np.sqrt(np.sum(eta_i**2))
            v_i = sign_vars[i] * eta_i.flatten() / den_i
            target_i = np.sum(v_i * y_M)
            resid_i = np.dot(I_n - np.outer(v_i,v_i), y_M)
            rho_i = np.dot(A, v_i)
            vec_i = (b.flat - np.dot(A, resid_i))/rho_i
            idx_neg_i = rho_i < 0
            idx_pos_i = rho_i > 0
            if idx_neg_i.any():
                vlo_i = np.max(vec_i[idx_neg_i])
            else:
                vlo_i = -np.inf
            if idx_pos_i.any():
                vup_i = np.min(vec_i[rho_i > 0])
            else:
                vup_i = +np.inf
            # Store
            targets[i], den[i] = target_i, den_i
            V_low[i], V_high[i] = vlo_i, vup_i
        # Transform the terms back to original scale
        eta2var = sigma2 * (den**2)
        v_neg = V_low * den
        v_pos = V_high * den
        # Return to original signs
        mask = sign_vars == -1
        V = np.c_[v_neg, v_pos]
        V[mask] = -V[mask][:,[1,0]]
        # Return terms
        return eta2var, V[:,0], V[:,1]


    def run_inference(self, alpha:float, null_beta:float | np.ndarray=0, sigma2:float | None=None, run_screen:bool=True, run_split:bool=True, run_carve:bool=True, run_ci:bool=True, **kwargs) -> None:
        """
        Carries out classical and PoSI inference (including data carving)
        
        Parameters
        ==========
        alpha:                  The type-I error rate
        null_beta:              The null hypothesis (default=0)
        sigma2:                 If estimate_sigma2() has not been run, user must specify the value
        run_screen:             Whether inference should be done on the truncated normal screened data (default=True)
        run_split:              Whether classical (normal/student-t) should be run of the split data (default=True)
        run_carve:              Whether carve infernce (NTS) should be run of screen+split data (default=True)
        run_ci:                 Whether CIs should be calculated
        **kwargs:               To be passed onto nts() dist

        Attributes
        ==========
        res_split:              Classical inference
        res_screen:             Selective inference with truncated normal
        res_carve:              Selective inference + classical = NTS
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
            sigma2_known = False
            assert hasattr(self, 'sig2hat'), 'if sigma2 is not specified, run estimate_sigma2 first'
            sigma2 = getattr(self, 'sig2hat')
            sigma2_ols = None
        else:
            sigma2_ols = sigma2
        assert isinstance(sigma2, float), 'sigma2 must be a float'

        # -- (ii) Calculate truncated normal for screened distribution -- #
        if run_screen:
            # Get the Truncated normal parameters
            eta2var, v_neg, v_pos = self._inference_on_screened(sigma2)
            dist_null = tnorm(null_beta, eta2var, v_neg, v_pos)
            # Assumes all values are positive
            bhat_M = self.ols_screen.linreg.coef_
            pval = dist_null.cdf(bhat_M)
            sign_M = np.sign(self.bhat_M)
            sign_neg = sign_M == -1
            pval = np.where(sign_neg, pval, 1-pval)
            self.res_screen = pd.DataFrame({'cidx':self.cidx_screen,'bhat':bhat_M, 'pval':pval})
            if run_ci:
                # Flip signs/axes where appropriate
                ci_lbub = dist_null.conf_int(bhat_M, alpha)
                self.res_screen['lb'] = ci_lbub[:,0]
                self.res_screen['ub'] = ci_lbub[:,1]

        # -- (iii) Calculate normal/student-t for screened distribution -- #
        if (self.frac_split > 0) and run_split:
            self.ols_split.run_inference(alpha=alpha, null_beta=null_beta, sigma2=sigma2_ols)
            self.res_split = self.ols_split.res_inf.copy()
            # Update the p-values to match the signs
            z = self.res_split['bhat'] / self.res_split['se']
            self.res_split['pval'] = np.where(sign_neg, Phi(z), Phi(-z))
            self.res_split.insert(0, 'cidx', self.cidx_screen)

        # -- (iv) Calculate NTS screen+split data (i.e. carving) -- #
        if run_carve:
            assert run_split and run_screen, 'if run_carved is to be run, run_{split,screen} must be True'
            mu1 = self.res_split['bhat'].copy()
            tau21 = self.res_split['se']**2
            mu2 = self.res_screen['bhat'].copy()
            tau22 = eta2var.copy()
            a, b = v_neg.copy(), v_pos.copy()
            n_A = len(self.x_split)
            n_B = len(self.x_screen)
            n = n_A + n_B
            c1 = n_A / n
            c2 = 1 - c1
            # Populate distribution under the null hypothesis
            bhat_carve = c1*mu1 + c2*mu2
            if 'cdf_approach' not in kwargs:
                cdf_approach = 'owen'
            else:
                cdf_approach = kwargs['cdf_approach']
            self.dist_carve = nts(null_beta, tau21, None, tau22, a, b, c1, c2, fix_mu=True, cdf_approach=cdf_approach)
            # Calculate terms for dataframe
            pval = self.dist_carve.cdf(bhat_carve)
            # Condition on the direction to test (otherwise screening will show more power unfairly)
            pval = np.where(sign_neg, pval, 1-pval)
            # pval = 2*np.minimum(pval, 1-pval)
            self.res_carve = pd.DataFrame({'cidx':self.cidx_screen, 'bhat':bhat_carve, 'pval':pval})
            if run_ci:
                ci_lbub = self.dist_carve.conf_int(bhat_carve, alpha=alpha, param_fixed='mu', cdf_approach= cdf_approach)
                ci_lbub = np.squeeze(ci_lbub)
                self.res_carve['lb'] = ci_lbub[:,0]
                self.res_carve['ub'] = ci_lbub[:,1]


        # Remove column used only for carve
        if run_split:
            self.res_split.drop(columns='se', errors='ignore', inplace=True)