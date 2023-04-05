"""
Utility scripts for simulations
"""
# Load modules
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
# Load internal modules
from utilities.utils import check_all_is_type

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
            check_all_is_type(float, )
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


def dgp_yX(n:int, p:int, s:int or None=None, b0:float or int=1, intercept:float or int=0, snr:float or int=1, seed:int=1, normalize:bool=True) -> tuple[np.ndarray, np.ndarray]:
    """Data generating process for simple gaussian-noise regression, where the columns are x are statistically independent
    
    Parameters
    ----------
    n:                  Number of samples
    p:                  Size of the design matrix
    s:                  Number of the non-zero coefficients
    b0:                 .....
    intercept:          ....
    snr:                Signal to nosie ratio ... (default=1)
    seed:               Reproducability seed (default=1)
    normalize:          Whether the design matrix should be normalized after creation (default=1)

    Returns
    -------
    An (x, y) tuple where x is a (n,p) design matrix, and y is an (n,) vector
    """
    # Input checks
    check_pos(strict=True, n, p, s, snr, seed)
    assert s <= p, 's cannot be larger than p'

    # Set up parameters
    beta = np.repeat(b0, p)
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


class two_stage():
    """Wrapper for performing a two-stage regerssio problem"""
    def __init__(self, n, m, gamma, alpha, pool=True, student=True):
        # Assign
        assert (n > 1) and (m >= 1) and (gamma > 0) & (gamma < 1)
        self.n, self.m, self.gamma = n, m, gamma
        self.alpha, self.pool = alpha, pool

        # Calculate degres of freedom
        self.dof_S, self.dof_T = n - 1, m - 1
        if self.pool:
            self.dof_T = n + m - 1
        if student:
            self.phi_inv = t(df=self.dof_S).ppf(1-gamma)
        else:
            self.phi_inv = norm.ppf(1-gamma)
        mn_ratio = m / n
        mu_2stage = np.array([0, -np.sqrt(mn_ratio)*self.phi_inv])
        tau_2stage = np.sqrt([1, mn_ratio])
        self.H0 = NTS(mu=mu_2stage,tau=tau_2stage, a=0, b=np.infty)
        self.HA = NTS(mu=mu_2stage,tau=tau_2stage, a=-np.infty, b=0)
        self.t_alpha = self.H0.ppf(alpha)
        self.power = self.HA.cdf(self.t_alpha)

    # self = dist_2s; nsim=100000; delta=2; sigma2=4; seed=None
    def rvs(self, nsim, delta, sigma2, seed=None):
        if seed is None:
            seed = nsim
        np.random.seed(seed)
        delta1 = delta + np.sqrt(sigma2/self.n)*np.random.randn(nsim)
        delta2 = delta + np.sqrt(sigma2/self.m)*np.random.randn(nsim)
        sigS = np.sqrt(sigma2*chi2(df=self.dof_S).rvs(nsim)/self.dof_S)
        sigT = np.sqrt(sigma2*chi2(df=self.dof_T).rvs(nsim)/self.dof_T)
        delta0 = delta1 + (sigS/np.sqrt(self.n))*self.phi_inv
        shat = (delta2 - delta0)/(sigT/np.sqrt(self.m))
        df = pd.DataFrame({'shat':shat, 'd0hat':delta0})
        return df


