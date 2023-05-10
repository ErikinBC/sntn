"""
Workhorse classes for doing marginal screening (posi_marginal_screen) and Lasso inference (posi_lasso)
"""

# External
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# Internal
from sntn.dists import tnorm, nts
from sntn._carve import _carve_yx
from sntn.utilities.linear import ols
from sntn.utilities.utils import get_valid_kwargs_cls


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
        


    def run_inference(self, alpha:float or np.ndarray=0.1, null_beta:float or np.ndarray=0, sigma2:float or None=None) -> pd.DataFrame:
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
        assert isinstance(alpha, float) or isinstance(alpha, np.ndarray), 'alpha must be a float or array'
        alpha = np.atleast_1d(alpha)
        assert np.all( (alpha > 0) & (alpha < 1) ), 'alpha must strictly be between (0,1)'
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
        A = None
        eta = np.linalg.inv(x_s.T.dot(x_s)).dot(x_s.T)  # (p, n) matrix
        Sigma = sigma2 * eta.T.dot(eta)  # Assume y ~ N(mu, sig2*I_n)
        alpha = A.dot(Sigma).dot(eta)
        breakpoint()


        # self.dist_screen = tnorm(mu=,sigma2=,a=,b=)

        # self.dist_carve = None
        # if self.frac_carve > 0:
        #     self.dist_carve = nts(mu1=,tau21=,mu2=,tau22=,a=,b=,)
        
        

