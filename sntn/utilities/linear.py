"""
Utility scripts for simulations
"""
# Load modules
import numpy as np
import pandas as pd
from math import isclose
from scipy.stats import norm, t
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
    def __init__(self, y:np.ndarray, x:np.ndarray, sigma2:None or float=None, has_int:bool=True) -> None:
        # Input checks
        assert isinstance(has_int, bool), 'has_int must be a bool'
        self.has_int = has_int
        if sigma2 is not None:
            check_all_is_type(sigma2, dtype=[float, int])
            check_all_pos(sigma2, strict=True)
        # Fit model with sklearn
        self.linreg = LinearRegression(fit_intercept=has_int)
        self.linreg.fit(X=x,y=y)
        self.n = len(x)
        self.k = x.shape[1] + has_int  # Number of parameters
        self.dof = self.n - self.k
        yhat = self.linreg.predict(x)
        self.sig2hat = np.sum((y - yhat)**2) / self.dof
        if self.has_int:
            iX = np.c_[np.repeat(1, self.n), x]
            self.igram = np.linalg.inv(iX.T.dot(iX))[1:,1:]
        else:
            self.igram = np.linalg.inv(x.T.dot(x))


    def run_inference(self, alpha:float, null_beta:float or np.ndarray=0, sigma2:float or None=None) -> None:
        """
        Carries out classical inference to get the p-values and CIs for linear regression

        Attributes
        ==========
        res_inf:            A DataFrame with columns 
        """
        # --- (i) Input checks --- #
        # variance of y ~ N(mu, sigma2*I_n)
        if sigma2 is None:
            student_t = True
            sigma2 = self.sig2hat
        else:
            student_t = False
            assert isinstance(sigma2, float) or isinstance(sigma2, int), 'if sigma2 is specified, should be float or int'
        # Null hypothesis for mu=X'null_beta
        if isinstance(null_beta, float) or isinstance(null_beta, int):
            null_beta = np.repeat(null_beta, self.igram.shape[0])
        if not isinstance(null_beta, np.ndarray):
            null_beta = np.asarray(null_beta)
        n_params_test = self.k-int(self.has_int)
        assert null_beta.shape[0] == n_params_test, f'Length of null_beta must {n_params_test} (accounts for {int(self.has_int)} intercept)'
        
        # Extract attributes
        bhat = self.linreg.coef_
        covar = sigma2 * self.igram
        se = np.sqrt(np.diagonal(covar))
        z = (bhat-null_beta) / se
        if student_t:
            dist = t(df=self.dof)
        else:
            dist = norm()
        crit_val = dist.ppf(1-alpha/2)
        pval = dist.cdf(z)
        pval = 2 * np.minimum(pval, 1-pval)
        lb = bhat - crit_val*se
        ub = bhat + crit_val*se
        self.res_inf = pd.DataFrame({'bhat':bhat, 'se':se, 'pval':pval, 'lb':lb, 'ub':ub})


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




