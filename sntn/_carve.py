"""
Workhorse classes for doing data carving
"""

# External
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
# Internal
from sntn.utilities.utils import cvec


class _carve_yx():
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
        # return normalize(mat, norm='l2', axis=0)
        return StandardScaler().fit_transform(mat)