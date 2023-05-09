"""
Utility scripts for simulations
"""
# Load modules
import numpy as np
from math import isclose
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
# Load internal modules
from sntn.utilities.utils import check_all_is_type, check_all_pos, cvec

class ols():
    """Ordinary least squares: model class wrapper

    Parameters
    ----------
    y:              An (n,) vector of response values
    x:              An (n,p) matrix of features
    sigma2:         Whether the error variance is known in advance (otherwise will be estimated)
    has_int:        Whether an intercept should be fit (default=True)
    alpha:          Type-I error rate for confidence intervals (default=0.05)
        
    Attributes
    ----------
    bhat:           (p,) covariate vector
    covar:          (p,p) covariance matrix (i.e. sigma2 * (X^T X)^-1)
    se:             (p,) marginal standard errors
    z:              (p,) marginal z-scores
    lb:             (p,) lower bounds of the CIs
    ub:             (p,) upper bounds of the CIs
    """
    def __init__(self, y:np.ndarray, x:np.ndarray, sigma2:None or float=None, has_int:bool=True, alpha:float=0.05, sig2:float or None=None) -> True:
        # Input checks
        if sigma2 is not None:
            check_all_is_type(sigma2, dtype=[float, int])
            check_all_pos(sigma2, strict=True)
        # Fit model with sklearn
        self.linreg = LinearRegression(fit_intercept=has_int)
        self.linreg.fit(X=x,y=y)
        self.n = len(x)
        self.k = x.shape[1] + has_int  # Number of parameters
        if sigma2 is None:
            yhat = self.linreg.predict(x)
            self.sig2hat = np.sum((y - yhat)**2) / (self.n - self.k)
        else:
            self.sig2hat = sigma2
        # Extract attributes
        self.bhat = self.linreg.coef_
        if has_int:
            self.bhat = np.append(np.array([self.linreg.intercept_]),self.bhat)
            iX = np.c_[np.repeat(1,self.n), x]
            self.igram = np.linalg.inv(iX.T.dot(iX))
        else:
            self.igram = np.linalg.inv(x.T.dot(x))        
        self.covar = self.sig2hat * self.igram
        self.se = np.sqrt(np.diagonal(self.covar))
        self.z = self.bhat / self.se
        # Calculate CIs
        cv = norm.ppf(1-alpha/2)
        self.lb = self.bhat - cv*self.se
        self.ub = self.bhat + cv*self.se


    def predict(self, x:np.ndarray) -> np.ndarray:
        """Wrapper for LinearRegression().predict()"""
        return self.linreg.predict(x)


def dgp_sparse_yX(n:int, p:int, s:int or None=None, beta:float or int or np.ndarray=None, intercept:float or int=0, snr:float or int=1, seed:int or None=1, return_params:bool=False) -> tuple[np.ndarray, np.ndarray]:
    """Data generating process for simple gaussian-noise regression, where the columns are x are statistically independent
    
    Parameters
    ----------
    n:                  Number of samples
    p:                  Size of the design matrix
    s:                  Number of the non-zero coefficients
    beta:               The coefficient that will get applied for every one of the first s
    intercept:          Whether an intercept should be added to X'beta
    snr:                Signal to nosie ratio, this will automatically lead to a certain sigma2 (default=1)
    seed:               Reproducability seed (default=1)
    return_params:      Whether beta,intercept should be returned as well

    Returns
    -------
    An (x, y) tuple where x is a (n,p) design matrix, and y is an (n,) vector
    """
    # Input checks
    check_all_pos(n, p, snr, strict=True)
    check_all_pos(s, strict=False)
    assert isinstance(return_params, bool), 'return_params needs to be a boolean'
    assert s <= p, 's cannot be larger than p'
    # Set up parameters
    beta0 = np.zeros(p)
    if isinstance(beta, np.ndarray):
        assert beta.shape == (p, ), f'if beta is manually assigned, must be a ({p},) array'
        beta0 = beta.copy()
    elif isinstance(beta, float) or isinstance(beta, int):
        # User has specified the non-sparse coefficients
        beta0[:s] = beta
    else:
        # Calculate the beta needed to get a certain signal to noise ratio (based on sigma2=1)
        beta0[:s] = np.sqrt(snr / s)
        assert isclose(np.sum(beta0**2), snr), 'snr did not align with expectation'
    # Generate data
    np.random.seed(seed)
    x = np.random.randn(n,p)
    x = (x - x.mean(0)) / x.std(0,ddof=1)  # Normalize
    u = np.random.randn(n)
    eta = x.dot(cvec(beta0)).flatten()
    y = intercept + eta + u
    if return_params:
        return y, x, intercept, beta0
    else:
        return y, x




