"""
Workhorse class for doing marginal screening 
"""

# External
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# Internal
from sntn.dists import tnorm, nts
from sntn._split import _split_yx
from sntn._cdf_bvn._utils import Phi
from sntn.utilities.linear import ols
from sntn.utilities.utils import get_valid_kwargs_cls, cvec, rvec


class _posi_marginal_screen(_split_yx):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(y, x, **get_valid_kwargs_cls(_split_yx, **kwargs))
        """
        Carries out selective inference for a marginal screening proceedure (i.e. top-k correlated features)

        Parameters
        ==========
        k:                  Number of top covariates to pick
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        **kwargs:           Passed into _split_yx(**kwargs) and ols(**kwargs)

        Methods
        =======
        estimate_sigma2:    Will combine unbiased "split" data estimate w/ CV on screened data
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
        
        # Fit an OLS model on the screened and split portion of the data
        ols_kwargs = get_valid_kwargs_cls(ols, **kwargs)
        self.ols_screen = ols(self.y_screen, self.x_screen[:,self.cidx_screen], **ols_kwargs)
        if self.frac_split > 0:
            self.ols_split = ols(self.y_split, self.x_split[:,self.cidx_screen], **ols_kwargs)


    def estimate_sigma2(self, num_folds:int=5) -> None:
        """
        Estimates the variance of residual of the marginally screened model. For the screened portion of the data, this amounts to doing k-fold CV

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
            tmp_model = self.__class__(k=self.k, y=y_train, x=x_train, frac_split=0.0)
            y_hat[tidx] = tmp_model.ols_screen.predict(x_test[:,tmp_model.cidx_screen])
        sig2hat_screen = np.mean((self.y_screen - y_hat)**2)
        
        # If carving exists, use it
        sig2hat_split = 0
        if self.frac_split > 0:
            sig2hat_split = self.ols_split.sig2hat
        
        # Use data-weighted final value
        self.sig2hat = (1-self.frac_split)*sig2hat_screen + self.frac_split*sig2hat_split
        

    def _get_A(self) -> np.ndarray:
        """
        Calculates the A from Ay <= b for the marginal screening proceedure
        
        Returns
        =======
        mat_A:          A (2*k*(p-k), n) matrix
        """
        # Create a (p,p) Identity matrix and remove the "selected" column rows
        partial = np.identity(self.p)[self.order[:-self.k]]
        partial = np.vstack([partial, -partial])  # (2*(p-k),p) matrix
        n_partial = partial.shape[0]
        # For each screened variable, another partial matrix will be added
        mat_A = np.zeros([partial.shape[0]*self.k, self.x_screen.shape[0]])
        for i, cidx in enumerate(self.cidx_screen[::-1]):
            partial_cp = partial.copy()
            partial_cp[:,cidx] = -self.s[cidx]
            start, stop = int(i*n_partial), int((i+1)*n_partial)
            mat_A[start:stop] = np.dot(partial_cp, self.x_screen.T)
        return mat_A
        

    def _inference_on_screened(self, sigma2:float | int) -> tuple:
        """
        Runs post selection inference for the screened variables

        Returns
        =======
        A tuple containing (alph_den, v_neg, v_pos) which is equivalent to the tnorm(; sigma2, a, b) terms
        """
        # See Lee (2014) for derivations of key terms
        mat_A = self._get_A()
        x_s = self.x_screen[:,self.cidx_screen]
        # assert all([isclose(xmu,0) for xmu in x_s.mean(0)]), 'Expected x_s to be de-meaned'
        # eta: The direction column span of x to be tested (k,n)
        bhat_S = self.ols_screen.linreg.coef_
        eta = np.dot(self.ols_screen.igram, x_s.T)
        # see Eq. 6 (note that sigma2 should cancel out)
        alph_num = sigma2 * np.dot(mat_A, eta.T)
        alph_den = sigma2 * np.sum(eta**2, axis=1)
        # A (q,k) array, where q is the number of constraints for each of the k dimensions being tested
        alph = alph_num / alph_den
        # mat_A is a (q,n) set of constaints on each of the dimensions of y
        # Ay is a (q,1) array, it is independent of the k coordinates
        Ay = np.dot(mat_A, cvec(self.y_screen))
        # For each column, find the alph's that are negative and apply Eq. 7, the alph's the are positive and apply Eq. 8
        ratio = -Ay/alph + rvec(bhat_S)  # alph_j*beta_j/alph_j = beta_j
        assert ratio.shape == alph.shape, 'ratio and alph should be the same shape'
        # Indexes are w.r.t to alph's
        v_neg = np.max(np.where(alph < 0, ratio, -np.inf), axis=0)
        v_pos = np.min(np.where(alph > 0, ratio, +np.inf), axis=0)
        assert np.all(v_pos > v_neg), 'expected pos to be larger than neg'
        return alph_den, v_neg, v_pos


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
            alph_den, v_neg, v_pos = self._inference_on_screened(sigma2)
            self.dist_screen = tnorm(null_beta, alph_den, v_neg, v_pos)
            # Add on the p-values and conf-int's
            bhat_S = self.ols_screen.linreg.coef_
            pval = self.dist_screen.cdf(bhat_S)
            sign_neg = self.s[self.cidx_screen] == -1
            pval = np.where(sign_neg, pval, 1-pval)
            self.res_screen = pd.DataFrame({'cidx':self.cidx_screen,'bhat':bhat_S, 'pval':pval})
            if run_ci:
                ci_lbub = self.dist_screen.conf_int(bhat_S, alpha, a=v_neg, b=v_pos, sigma2=alph_den)
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
            tau22 = alph_den.copy()
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
            self.dist_carve = nts(null_beta, tau21, None, tau22, a, b, c1, c2, cdf_approach=cdf_approach, fix_mu=True)
            # Calculate terms for dataframe
            pval = self.dist_carve.cdf(bhat_carve)
            # Condition on the direction to test (otherwise screening will show more power unfairly)
            pval = np.where(sign_neg, pval, 1-pval)
            self.res_carve = pd.DataFrame({'cidx':self.cidx_screen, 'bhat':bhat_carve, 'pval':pval})
            if run_ci:
                ci_lbub = self.dist_carve.conf_int(bhat_carve, alpha=alpha, param_fixed='mu', cdf_approach=cdf_approach)
                ci_lbub = np.squeeze(ci_lbub)
                self.res_carve['lb'] = ci_lbub[:,0]
                self.res_carve['ub'] = ci_lbub[:,1]

        # Remove column used only for carve
        if run_split:
            self.res_split.drop(columns='se', errors='ignore', inplace=True)
        

