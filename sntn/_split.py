"""
Workhorse class for splitting data to do data carving/unbiased inference
"""

# External
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
# Internal
from sntn.utilities.utils import cvec


class _split_yx():
    def __init__(self, y:np.ndarray, x:np.ndarray, frac_split:float=0.5, seed:int or None=None, normalize:bool=True, has_int:bool=True) -> None:
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
        assert isinstance(frac_split, float), 'frac_split needs to be a float'
        assert (frac_split >= 0) and (frac_split < 1), 'frac_split must be between [0,1)'
        assert isinstance(has_int, bool), 'has_int must be a bool'
        self.normalize = normalize
        self.frac_split = frac_split
        self.has_int = has_int
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
        self.ridx_split = np.array([], dtype=int)
        self.x_split, self.y_split = self.ridx_split.copy(), self.ridx_split.copy()
        self.ridx_screen = np.arange(self.n)
        if self.frac_split > 0:
            n_split = int(np.floor(self.n * self.frac_split))
            n_screen = self.n - n_split
            # Select rows to be used for data carving
            np.random.seed(self.seed)
            self.ridx_split = np.random.choice(self.n, n_split, replace=False)
            self.ridx_screen = np.setdiff1d(self.ridx_screen, self.ridx_split)
        # Assign the screening arrays
        self.x_split = x[self.ridx_split]
        self.x_screen = x[self.ridx_screen]
        self.y_split = y[self.ridx_split]
        self.y_screen = y[self.ridx_screen]

        # Normalize the matrices if requested
        if normalize:
            self.x_screen = self.normalize_mat(self.x_screen)
            if self.frac_split > 0:
                self.x_split = self.normalize_mat(self.x_split)

    @staticmethod
    def normalize_mat(mat:np.ndarray) -> np.ndarray:
        # return normalize(mat, norm='l2', axis=0)
        return StandardScaler().fit_transform(mat)