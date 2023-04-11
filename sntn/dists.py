"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
from copy import deepcopy
from scipy.stats import truncnorm, norm
from scipy.optimize import root, minimize_scalar, minimize, root_scalar
# Internal modules
from sntn.utilities.utils import broastcast_max_shape, grad_clip_abs
from sntn.utilities.grad import _log_gauss_approx, _log_diff


class tnorm():
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int) -> None:
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


    @staticmethod
    def _err_cdf(mu:np.ndarray, x:np.ndarray, sigma:np.ndarray, a:np.ndarray, b:np.ndarray, alpha:float, approx:bool=False, a_min=None, a_max=None) -> np.ndarray:
        """
        Internal method to feed into the root findings/minimizers. Assumes that sigma/a/b are constant
        
        Parameters
        ----------
        mu:             A candidate array of means (what is being optimized over)
        x:              The observed data points
        sigma:          Standard deviation
        a:              Lowerbound
        b:              Upperbound
        alpha:          The desired CDF level
        approx:         Whether the norm.{cdf,pdf} should be used or the _log_gauss_methods for tail probabilities
        a_min:          Whether lowerbound clipping should be applied to the gradient
        a_max:          Whether upperbound clipping should be applied to the gradient
        """
        a_trans = (a-mu)/sigma
        b_trans = (b-mu)/sigma
        dist = truncnorm(loc=mu, scale=sigma, a=a_trans, b=b_trans)
        err = dist.cdf(x) - alpha
        return err

    def _err_cdf0(self, mu, x, sigma, a, b, alpha, approx, a_min, a_max) -> float:
        """Returns the 1, array as a float"""
        res = float(self._err_cdf(mu, x, sigma, a, b, alpha, approx, a_min, a_max))
        return res

    def _err_cdf2(self, mu, x, sigma, a, b, alpha, approx, a_min, a_max) -> np.ndarray:
        """Call _err_cdf and returns the square of the results"""
        err = self._err_cdf(mu, x, sigma, a, b, alpha, approx, a_min, a_max)
        err2 = np.sum(err**2)
        return err2

    @staticmethod
    def _dmu_dcdf(mu, x, sigma, a, b, alpha, approx, a_min, a_max) -> np.ndarray:
        """
        For a given mean/point, determine the derivative of the CDF w.r.t. the location parameter (used by numerical optimziation soldres)
        
        dF(mu)/du = {[(phi(alpha)-phi(xi))/sigma]*z + [(phi(alpha)-phi(beta))/sigma]*[Phi(xi)-Phi(alpha)] } /z^2
        z = Phi(beta) - Phi(alpha)

        Parameters
        ----------
        see 
        """
        # Normalize the inputs
        x_z = (x-mu)/sigma
        a_z = (a-mu)/sigma
        b_z = (b-mu)/sigma
        # Calculate derivatives from one of two approaches
        if approx:
            log_term1a = _log_gauss_approx(x_z, a_z, False) - np.log(sigma)
            log_term1b = _log_gauss_approx(b_z, a_z, True)
            log_term2a = _log_gauss_approx(x_z, a_z, True)
            log_term2b = _log_gauss_approx(b_z, a_z, False) - np.log(sigma)
            # The numerator will be off when the terms 1/2 do not align
            sign1a = np.where(np.abs(a_z) > np.abs(x_z),-1,+1)
            sign1b = np.where(b_z > a_z,+1,-1)
            term1_signs = sign1a * sign1b
            sign2a = np.where(x_z > a_z,+1,-1)
            sign2b = np.where(np.abs(a_z) > np.abs(b_z),-1,+1)
            term2_signs = sign2a * sign2b
            sign_fail = np.where(term1_signs != term2_signs, -1, +1) 
            log_fgprime = log_term1a + log_term1b
            log_gfprime = log_term2a + log_term2b
            log_num = np.real(_log_diff(log_fgprime, log_gfprime, sign_fail))
            log_denom = 2*_log_gauss_approx(b_z, a_z,True)
            dFdmu = -np.exp(log_num - log_denom)
            # breakpoint()
            dFdmu = grad_clip_abs(dFdmu, a_min, a_max)
        else:
            term1a = (norm.pdf(a_z)-norm.pdf(x_z))/sigma
            term1b = norm.cdf(b_z) - norm.cdf(a_z)
            term2a = norm.cdf(x_z) - norm.cdf(a_z)
            term2b = (norm.pdf(a_z)-norm.pdf(b_z))/sigma
            # quotient rule
            dFdmu = (term1a*term1b - term2a*term2b)/term1b**2  
        # print(f'mu={mu},val={dFdmu}')
        # if np.any(np.isnan(dFdmu)):
        return dFdmu

    def _derr_cdf2(self, mu, x, sigma, a, b, alpha, approx, a_min, a_max) -> np.ndarray:
        """Wrapper for the derivative of d/dmu (F(mu) - alpha)**2 = 2*(F(mu)-alpha)*(d/dmu F(mu))"""
        term1 = 2*self._err_cdf(mu, x, sigma, a, b, alpha, approx, a_min, a_max)
        term2 = self._dmu_dcdf(mu, x, sigma, a, b, alpha, approx, a_min, a_max)
        res = term1 * term2
        return res


    def get_CI(self, x:np.ndarray, approach:str, alpha:float=0.05, approx:bool=True, a_min:None or float=0.005, a_max:None or float=None, mu_lb:float or int=-100000, mu_ub:float or int=100000, **kwargs) -> np.ndarray:
        """
        Assume X ~ TN(mu, sigma, a, b), where sigma, a, & b are known. Then for any realized mu_hat (which can be estimated with self.fit()), we want to find 
        Calculate the confidence interval for a series of points (x)

        Parameters
        ----------
        x:                  An array-like object of points that corresponds to dimensions of estimated means
        approach:           A total of four approaches have been implemented to calculate the CIs (see scipy.optimize.{root, minimize_scalar, minimize, root_scalar})
        alpha:              Type-I error for the CIs (default=0.05)
        mu_lb:              Found bounded optimization methods, what is the lower-bound of means that will be considered for the lower-bound CI?
        mu_ub:              Found bounded optimization methods, what is the upper-bound of means that will be considered for the lower-bound CI?
        kwargs:             Named arguments which will be passed into the scipy.optims

        Returns
        -------
        An ({x.shape,mu.shape},2) array for the lower/upper bound. Shape be different than x if x gets broadcasted by the existing parameters
        """
        # Input checks
        valid_approaches = ['root', 'minimize_scalar', 'minimize', 'root_scalar']
        assert approach in valid_approaches, f'approach needs to be one of {valid_approaches}'
        # Try to broadcast x to match underlying parameters
        # Guess some lower/upper bounds
        c_alpha = norm.ppf(alpha/2)
        x0_lb, x0_ub = self.mu + c_alpha, self.mu - c_alpha
        # Squeeze x if we can in cause parameters are (n,)
        x = np.squeeze(x)
        x, mu, sigma, a, b, x0_lb, x0_ub = broastcast_max_shape(x, self.mu, self.sigma, self.a, self.b, x0_lb, x0_ub)
        
        # Set a default "method" for each if not supplied in the kwargs
        di_default_methods = {'root':'hybr',
                              'minimize_scalar':'Brent',
                              'minimize':'COBYLA',
                              'root_scalar':'bisect'}
        if 'method' not in kwargs:
            kwargs['method'] = di_default_methods[approach]

        # Set up the argument for the vectorized vs scalar methods
        if approach in ['minimize_scalar','root_scalar']:
            # x, sigma, a, b, alpha, approx, a_min, a_max
            solver_args = [x.flat[0], self.sigma.flat[0], self.a.flat[0], self.b.flat[0], 1-alpha/2, approx, a_min, a_max]
            # Will be assigned iteratively
            ci_lb = np.full_like(mu, fill_value=np.nan)
            ci_ub = ci_lb.copy()
        else:
            solver_args = [x, self.sigma, self.a, self.b, alpha, approx, a_min, a_max]        

        if approach == 'root_scalar':
            # ---- Approach #1: Point-wise root finding ---- #
            # There are four different approaches to configure root_scalar (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
            di_root_scalar_extra = {}
            if kwargs['method'] in ['bisect', 'brentq', 'brenth', 'ridder','toms748']:
                di_root_scalar_extra['bracket'] = (mu_lb, mu_ub)
            elif kwargs['method'] == 'newton':
                di_root_scalar_extra['x0'] = None
                di_root_scalar_extra['fprime'] = self._dmu_dcdf
            elif kwargs['method'] == 'secant':
                di_root_scalar_extra['x0'] = None
                di_root_scalar_extra['x1'] = None
            elif kwargs['method'] == 'halley':
                di_root_scalar_extra['x0'] = None
                di_root_scalar_extra['fprime'] = self._dmu_dcdf
                di_root_scalar_extra['fprime2'] = self._dmu_dcdf

            # Prepare the parts of the optimization that won't change (note that kwargs will overwrite di_base)
            di_base = {'f':self._err_cdf0, 'xtol':1e-4, 'maxiter':250}  # Hard-coding unless specified by user based on stability experiments
            di_base = {**di_base, **kwargs}
            # Loop over all element points
            for kk in np.ndindex(x.shape): 
                # Update the solver_args
                mu_kk, x_kk, sigma_kk, a_kk, b_kk = mu[kk], x[kk], sigma[kk], a[kk], b[kk]
                solver_args[:4] = x_kk, sigma_kk, a_kk, b_kk
                # Extra kk'th element
                if 'x0' in di_root_scalar_extra:
                    di_root_scalar_extra['x0'] = mu_kk
                if 'x1' in di_root_scalar_extra:
                    di_root_scalar_extra['x1'] = x_kk
                # Hard-coding iterations
                di_lb = {**di_base, **{'args':solver_args}, **di_root_scalar_extra}
                di_ub = deepcopy(di_lb)
                di_ub['args'][4] = alpha/2
                di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
                lb_kk = float(root_scalar(**di_lb).root)
                ub_kk = float(root_scalar(**di_ub).root)
                ci_lb[kk] = lb_kk
                ci_ub[kk] = ub_kk            

        elif approach == 'minimize_scalar':
            # ---- Approach #2: Point-wise gradient ---- #
            
            di_base = {'fun':self._err_cdf2, 'bounds':(mu_lb, mu_ub)}
            di_base = {**di_base, **kwargs}
            # Loop over all element points
            for kk in np.ndindex(x.shape):
                mu_kk, x_kk, sigma_kk, a_kk, b_kk = mu[kk], x[kk], sigma[kk], a[kk], b[kk]
                solver_args[:4] = x_kk, sigma_kk, a_kk, b_kk
                di_lb = {**di_base, **{'args':solver_args}}
                di_ub = deepcopy(di_lb)
                di_ub['args'][4] = alpha/2
                di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
                lb_kk = minimize_scalar(**di_lb).x
                ub_kk = minimize_scalar(**di_ub).x
                ci_lb[kk] = lb_kk
                ci_ub[kk] = ub_kk
        
        elif approach == 'minimize':
            # ---- Approach #3: Vector gradient ---- #
            di_base = {**{'fun':self._err_cdf2, 'x0':mu},**kwargs}
            ci_lb = minimize(**{**di_base, **{'args':(x, 1-alpha/2, approx, a_min, a_max)}}).x
            ci_ub = minimize(**{**di_base, **{'args':(x, alpha/2, approx, a_min, a_max)}}).x
        else:
            # ---- Approach #4: Gradient root finding ---- #
            di_lb = {**{'fun':self._err_cdf, 'x0':x0_lb, 'args':(x, 1-alpha/2, approx, a_min, a_max)},**kwargs}
            di_ub = {**{'fun':self._err_cdf, 'x0':x0_ub, 'args':(x, alpha/2, approx, a_min, a_max)},**kwargs}
            ci_lb = root(**di_lb).x
            ci_ub = root(**di_ub).x
        # Return values
        mat = np.stack((ci_lb,ci_ub),ci_lb.ndim)
        return mat 