"""
Workhorse classes for doing marginal screening (posi_marginal_screen) and Lasso inference (posi_lasso)
"""

# External
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# Internal
from sntn.dists import tnorm, nts
from sntn.utilities.linear import ols
from sntn.utilities.utils import cvec, get_valid_kwargs_cls


class _data_carve():
    def __init__(self, y:np.ndarray, x:np.ndarray, frac_carve:float=0.5, seed:int or None=None, normalize:bool=True, has_int:bool=True) -> None:
        """
        Class which

        Parameters
        ==========
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        frac_carve:         What percent of the data should be used for data carving? (0==no carving, ->1 all inference (noisy selection))? If frac_carve >0, it must have at least p+2 observations
        normalize:          Whether X should be normalized (defaul=True)
        has_int:            Whether an intercept should be fit for the OLS model (default=True)
        
        Attributes
        ==========
        {y, x, frac_carv, normalize, has_int}
        ridx_carve:          Which rows will be used for carving?
        ridx_screen:         Which rows will be used for screening?

        """
                # Input checks
        assert len(y) == len(x), 'length of y and x do not align'
        assert isinstance(normalize, bool), 'normalize needs to be a bool'
        assert isinstance(frac_carve, float), 'frac_carve needs to be a float'
        assert (frac_carve >= 0) and (frac_carve < 1), 'frac_carve must be between [0,1)'
        assert isinstance(has_int, bool), 'has_int must be a bool'
        self.normalize = True
        self.frac_carve = frac_carve
        self.has_int = True
        self.seed = seed

        # Process response and covariates
        self.cn = None
        if isinstance(x, pd.DataFrame):
            self.cn = x.columns
            x = np.asarray(x)
        else:
            x = cvec(x)
            self.cn = [f'x{i+1}' for i in range(x.shape[1])]
        assert len(x.shape) == 2, 'Expected x to be (n,p)'
        y = np.asarray(y)
        self.n, self.p = x.shape
        self.enc_x = None
        
        # Split the data for data carving
        self.ridx_carve = np.array([], dtype=int)
        self.x_carve, self.y_carve = self.ridx_carve.copy(), self.ridx_carve.copy()
        self.ridx_screen = np.arange(self.n)
        if self.frac_carve > 0:
            n_carve = int(np.floor(self.n * self.frac_carve))
            n_screen = self.n - n_carve
            # Select rows to be used for data carving
            np.random.seed(self.seed)
            self.ridx_carve = np.random.choice(self.n, n_carve, replace=False)
            self.ridx_screen = np.setdiff1d(self.ridx_screen, self.ridx_carve)
        # Assign the screening arrays
        self.x_carve = x[self.ridx_carve]
        self.x_screen = x[self.ridx_screen]
        self.y_carve = y[self.ridx_carve]
        self.y_screen = y[self.ridx_screen]

        # Normalize the matrices if requested
        if normalize:
            self.x_screen = self.normalize_mat(self.x_screen)
            if self.frac_carve > 0:
                self.x_carve = self.normalize_mat(self.x_carve)

    @staticmethod
    def normalize_mat(mat:np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(mat)



class _posi_marginal_screen(_data_carve):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(y, x, **get_valid_kwargs_cls(_data_carve, **kwargs))
        """
        Carries out selective inference for a marginal screening proceedure (i.e. top-k correlated features)

        Parameters
        ==========
        k:                  Number of top covariates to pick
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        **kwargs:           Passed into _data_carve(**kwargs) and ols(**kwargs)


        Methods
        =======
        estimate_sigma2:    Can use "carved" data to estimate the variance of the model (otherwise will need to be specified)
        run_inference:      Carries out post-selection inference on data
        """
        # Input checks
        assert isinstance(k, int) and (k > 0), 'k must be an int that is greater than zero'
        assert k < self.p, f'For screening to be relevant, k must be less than {self.p}'
        self.k = k

        # Use the "screening" portion of the data to find the top-K coefficients
        beta_screen = pd.Series(np.abs(self.x_screen.T.dot(self.y_screen)))
        beta_screen = beta_screen.sort_values(ascending=False)
        self.cidx_screen = beta_screen.head(self.k).index.to_list()

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
        

    def run_inference(self, alpha:float=0.1, null_beta:float=0, sigma2:float or None=None) -> pd.DataFrame:
        """
        Carries out inferences...
        
        Parameters
        ==========
        alpha:                  The type-I error rate
        null_beta:              The null hypothesis (default=0)
        sigma2:                 If estimate_sigma2() has not been run, user must specify the value
        """
        if sigma2 is None:
            assert hasattr(self, 'sigma2'), 'if sigma2 is not specified, run estimate_sigma2 first'
            sigma2 = getattr(self, sigma2)
        if self.frac_carve == 0:
            dist
            tnorm(mu, sigma2, a, b)
        return None
        

