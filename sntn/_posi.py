"""
Workhorse classes for doing marginal screening (posi_marginal_screen) and Lasso inference (posi_lasso)
"""

# External
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# Internal
from sntn.dists import tnorm, nts
from sntn.utilities.utils import cvec
from sntn.utilities.linear import ols


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
            np.random.seed(seed)
            self.ridx_carve = np.random.choice(self.n, n_carve, replace=False)
            self.ridx_screen = np.setdiff1d(self.ridx_screen, self.ridx_carve)
        # Assign the screening arrays
        self.x_carve = x[self.ridx_carve]
        self.x_screen = x[self.ridx_screen]
        self.y_carve = y[self.ridx_carve]
        self.y_screen = y[self.ridx_screen]

        # Normalize the matrices if requested
        if normalize:
            self.x_carve = self.normalize_mat(self.x_carve)
            self.x_screen = self.normalize_mat(self.x_screen)

    @staticmethod
    def normalize_mat(mat:np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(mat)



class _posi_marginal_screen(_data_carve):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(y, x, **kwargs)
        """
        Carries out selective inference for a marginal screening proceedure (i.e. top-k correlated features)

        Parameters
        ==========
        k:                  Number of top covariates to pick
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        **kwargs:           See _data_carve


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
        breakpoint()
        


    def estimate_sigma2(self) -> None:
        """Words"""
        if self.frac_carve > 0:
            1

    
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
        

n, p = 50, 100
b0, s,  = -1, 5
snr = 2
from sntn.utilities.linear import dgp_sparse_yX
y, x, beta0, beta1 = dgp_sparse_yX(n, p, s, intercept=b0, snr=snr, return_params=True)

# Initialize
k = 5
screener = _posi_marginal_screen(k, y, x)
