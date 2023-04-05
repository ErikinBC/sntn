"""
Utility scripts for simulations
"""
# Load modules
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
# Load internal modules
from utilities.utils import check_all_is_type, check_all_pos

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
    def __init__(self, y:np.ndarray, x:np.ndarray, sigma2:None or float=None, has_int:bool=True, alpha:float=0.05) -> True:
        # Input checks
        if sigma2 is not None:
            check_all_is_type(sigma2, [float, int])
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
            gram = np.linalg.inv(iX.T.dot(iX))
        else:
            gram = np.linalg.inv(x.T.dot(x))
        self.covar = self.sig2hat * gram
        self.se = np.sqrt(np.diagonal(self.covar))
        self.z = self.bhat / self.se
        # Calculate CIs
        cv = norm.ppf(1-alpha/2)
        self.lb = self.bhat - cv*self.se
        self.ub = self.bhat + cv*self.se


def dgp_sparse_yX(n:int, p:int, s:int or None=None, beta:float or int or np.ndarray=1, intercept:float or int=0, snr:float or int=1, seed:int=1, normalize:bool=True) -> tuple[np.ndarray, np.ndarray]:
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
    normalize:          Whether the design matrix should be normalized after creation (default=1)

    Returns
    -------
    An (x, y) tuple where x is a (n,p) design matrix, and y is an (n,) vector
    """
    # Input checks
    check_all_pos(n, p, snr, strict=True)
    check_all_pos(s, seed, strict=False)
    assert isinstance(normalize, bool), 'normalize needs to be a boolean'
    assert s <= p, 's cannot be larger than p'
    # Set up parameters
    if not isinstance(beta, np.ndarray):
        beta = np.repeat(beta, p)
    var_exp = np.sum(beta**2)
    sig2 = 1
    if var_exp > 0:
        sig2 =  var_exp / snr
    # Generate data
    np.random.seed(seed)
    x = np.random.randn(n,p)
    if normalize:
        x = (x - x.mean(0)) / x.std(0,ddof=1)
    u = np.sqrt(sig2)*np.random.randn(n)
    eta = x.dot(beta)
    y = intercept + eta + u
    return y, x




