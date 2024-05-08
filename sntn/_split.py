"""
Workhorse class for splitting data to do data carving/unbiased inference
"""

# External
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Internal
from sntn.utilities.utils import cvec


class _split_yx():
    def __init__(self, y:np.ndarray, x:np.ndarray, frac_split:float=0.5, seed:int | None=None, normalize:bool=True, has_int:bool=True) -> None:
        """
        Class which

        Parameters
        ==========
        y:                  (n,) array of responses
        x:                  (n,p) array of covariates
        frac_split:         What percent of the data should be used for data carving? (0==no carving, ->1 all inference (noisy selection))? If frac_split >0, it must have at least p+2 observations
        normalize:          Whether X should be normalized (defaul=True)
        has_int:            Whether an intercept should be fit for the OLS model (default=True)
        
        Attributes
        ==========
        {y, x, frac_carv, normalize, has_int}
        ridx_split:          Which rows will be used for carving?
        ridx_screen:         Which rows will be used for screening?

        """
                # Input checks
        assert len(y) == len(x), 'length of y and x do not align'
        assert isinstance(normalize, bool), 'normalize needs to be a bool'
        assert isinstance(frac_split, float) or isinstance(frac_split, int), 'frac_split needs to be a float or an int'
        assert (frac_split >= 0) and (frac_split < 1), 'frac_split must be between [0,1)'
        assert isinstance(has_int, bool), 'has_int must be a bool'
        self.normalize = normalize
        self.frac_split = frac_split
        self.has_int = has_int
        self.seed = seed
        self.has_split = self.frac_split > 0

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
        self.holder = np.array([], dtype=int)
        if self.has_split:
            self.x_screen, self.x_split, self.y_screen, self.y_split = train_test_split(x, y, test_size=frac_split, random_state=seed)
        else:
            self.x_screen, self.y_screen = x, y
            self.x_split, self.y_split = self.holder.copy(), self.holder.copy()
        # Normalize the matrices if requested
        if normalize:
            self.x_screen, mu_screen, se_screen = self.normalize_mat(self.x_screen, return_mu_se=True)
            if self.has_split:
                mu_split = self.x_split.mean(0)
                self.x_split = (self.x_split - mu_split) / se_screen


    @staticmethod
    def normalize_mat(mat:np.ndarray, mu:None | np.ndarray=None, se:None | np.ndarray=None, return_mu_se:bool = False) -> np.ndarray:
        if mu is None:
            mu = mat.mean(axis=0)
        if se is None:
            se = mat.std(axis=0, ddof=1)
        res = (mat - mu) / se
        if return_mu_se:
            return res, mu, se
        else:
            return res
