"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
from scipy.stats import truncnorm
# Internal modules
from sntn.utilities.utils import vprint, broastcast_max_shape


class tnorm():
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int, verbose:bool=False) -> None:
        """
        Main model class for the truncated normal distribution

        Parameters
        ----------
        mu:                 Means of the unconditional normal (can be array)
        sigma2:             Variance of the truncated normal (can be array)
        a:                  Lower bounds of truncated normal (can be -infty)
        b:                  Upperbounds of truncated normal (can be +infty)
        verbose:            Whether printing will occur some methods

        Attributes
        ----------
        mu:                 Array of means
        sigma:              Array of standard deviations
        a:                  Array of lowerbounds
        b:                  Array of upperbounds
        alpha:              Array of z-score lowerbounds
        beta:               Array of z-score upperbounds
        dist:               The scipy truncnorm

        Methods
        -------
        ....
        """
        # Input checks and broadcasting
        self.mu, sigma2, self.a, self.b = broastcast_max_shape(mu, sigma2, a, b)
        assert np.all(self.a < self.b), 'a needs to be < b'
        assert np.all(sigma2 > 0), 'sigma2 needs to be > 0'
        self.sigma = np.sqrt(sigma2)
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        # Calculate the dimension size
        self.param_shape = self.mu.shape
        # Initialize the distribution
        self.dist = truncnorm(loc=self.mu, scale=self.sigma, a=self.alpha, b=self.beta)

    def cdf(self, x:np.ndarray) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).cdf()"""
        return self.dist.cdf(x)

    def ppf(self, x:np.ndarray) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).ppf()"""
        return self.dist.ppf(x)

    def pdf(self, x:np.ndarray) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).pdf()"""
        return self.dist.pdf(x)

    def fit(self, x:np.ndarray, use_a:bool=True, use_b:bool=True, use_sigma:bool=False) -> tuple:
        """
        Wrapper for the fit argument (assumes a/b are constant). Order is: a,b,mu,sigma

        Parameters
        ----------
        x:              Data where the first axis is the samples (i.e. x[:,0,0,..] should be of len(nsamp))
        use_a:          Whether the lowerbound should be used from initialization
        use_b:          Whether the upperbound should be used from initialization
        use_sigma:      Whether the variance should be used from initialization

        Returns
        -------
        A tuple of length four, with estimates of (a_hat, b_hat, mu_hat, sigma_hat). Note that if usa_{a,b,sigma} is provided the _hat values will be the same as the values set during initialization
        """
        # Input prep
        ns_x = len(x.shape)
        fa, fb, fsigma = None, None, None
        # Solve for vector case
        if ns_x == 1:  # No need to loop since x is just an array
            if use_a:
                fa = self.alpha
            if use_b:
                fb = self.beta
            if use_sigma:
                fsigma = self.sigma
            di = dict(zip(['fa','fb','fscale'],[fa,fb,fsigma]))
            di = {k:v for k,v in di.items() if v is not None}
            a_hat, b_hat, mu_hat, sigma_hat = truncnorm.fit(x, **di)
        else:
            # Solve for matrix case
            a_hat = self.mu * np.nan
            b_hat, mu_hat, sigma_hat = a_hat.copy(), a_hat.copy(), a_hat.copy()
            # Loop over all dimensions except the first
            for kk in np.ndindex(x.shape[1:]):
                x_kk = x[np.s_[:,] + kk]  # Extract as array
                if use_a:
                    fa = self.alpha[kk]
                if use_b:
                    fb = self.beta[kk]
                if use_sigma:
                    fsigma = self.sigma[kk]
                di = dict(zip(['fa','fb','fscale'],[fa,fb,fsigma]))
                di = {k:v for k,v in di.items() if v is not None}
                res_kk = truncnorm.fit(x_kk, **di)
                a_hat[kk], b_hat[kk], mu_hat[kk], sigma_hat[kk] = res_kk
        return a_hat, b_hat, mu_hat, sigma_hat


    def rvs(self, n, seed=None) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).rvs()"""
        # When sampling it is [num_sample,*dims of parameters]
        samp_shape = (n,)
        if self.mu.shape != (1,):  # If everything is a float, no need for a second dimension
            samp_shape += self.mu.shape
        return self.dist.rvs(samp_shape, random_state=seed)


    # approach='root';verbose=True;alpha=0.05;kwargs={}
    def get_CI(self, x:np.ndarray, approach:str, alpha:float=0.05, verbose:bool=False, **kwargs):
        """
        Assume X ~ TN(mu, sigma, a, b), where sigma, a, & b are known. Then for any realized mu_hat (which can be estimated with self.fit()), we want to find 
        Calculate the confidence interval for a series of points (x)

        Parameters
        ----------
        x:                  An array-like object of points that corresponds to dimensions of estimated means
        approach:           A total of four approaches have been implemented to calculate the CIs (see scipy.optimize.{root, minimize_scalar, minimize, root_scalar})

        kwargs:             Named arguments which will be passed into the scipy.optims

        Returns
        -------
        """
        from scipy.optimize import root, minimize_scalar, minimize, root_scalar
        # Input checks
        valid_approaches = ['root', 'minimize_scalar', 'minimize', 'root_scalar']
        assert approach in valid_approaches, f'approach needs to be one of {valid_approaches}'
        
        # Try to broadcast x to match underlying parameters
        x, _ = broastcast_max_shape(x, self.mu)
        # Define the optimization function
        from scipy import optimize
        optim_fun = getattr(optimize, approach)
        
        if approach == 'root_scale':
            print('root_scaler')
        elif approach == 'minimize_scaler':
            print('minimize_scaler')
        elif approach == 'minimize':
            print('minimize')
        else:
            print('root')

        # Run optimize
        if len(kwargs) > 0:
            res = optim_fun(1, **kwargs)
        else:
            res = optim_fun(1)
        
        


    #     k = max(self.p, x.shape[1])
    #     # Initialize
    #     mu_seq = np.round(np.sinh(np.linspace(np.repeat(np.arcsinh(lb),k), 
    #         np.repeat(np.arcsinh(ub),k),nline)),5)
    #     q_seq = np.zeros(mu_seq.shape)
    #     q_err = q_seq.copy()
    #     cidx = list(range(k))
    #     iactive = cidx.copy()
    #     pidx = cidx.copy()
    #     j, aerr = 0, 1
    #     while (j<=imax) & (len(iactive)>0):
    #         j += 1
    #         vprint('------- %i -------' % j, verbose)
    #         # Calculate quantile range
    #         mus = mu_seq[:,iactive]
    #         if len(iactive) == 1:
    #             mus = mus.flatten()
    #         if self.p == 1:
    #             pidx = np.repeat(0, len(iactive))
    #         elif len(iactive)==1:
    #             pidx = iactive
    #         else:
    #             pidx = iactive
    #         qs = tnorm(mus, self.sig2[pidx], self.a[pidx], self.b[pidx]).ppf(gamma)
    #         if len(qs.shape) == 1:
    #             qs = cvec(qs)
    #         q_seq[:,iactive] = qs
    #         tmp_err = q_seq - x
    #         q_err[:,iactive] = tmp_err[:,iactive]
    #         istar = np.argmin(q_err**2,0)
    #         q_star = q_err[istar, cidx]
    #         mu_star = mu_seq[istar,cidx]
    #         idx_edge = (mu_star == lb) | (mu_star == ub)
    #         aerr = 100*np.abs(q_star)
    #         if len(aerr[~idx_edge]) > 0:
    #             vprint('Largest error: %0.4f' % max(aerr[~idx_edge]),verbose)
    #         idx_tol = aerr < tol
    #         iactive = np.where(~(idx_tol | idx_edge))[0]
    #         # Get new range
    #         if len(iactive) > 0:
    #             new_mu = np.linspace(mu_seq[np.maximum(0,istar-1),cidx],
    #                                 mu_seq[np.minimum(istar+1,nline-1),cidx],nline)
    #             mu_seq[:,iactive] = new_mu[:,iactive]
    #     mu_star = mu_seq[istar, cidx]
    #     # tnorm(mu=mu_star, sig2=self.sig2, a=self.a, b=self.b).ppf(gamma)
    #     return mu_star


