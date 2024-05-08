"""
Contains the raw methods that get wrapped in dists.py
"""

# External
import numpy as np
from scipy.stats import truncnorm, norm
# Internal
from sntn.utilities.grad import _log_gauss_approx, _log_diff
from sntn.utilities.utils import broastcast_max_shape, grad_clip_abs
from sntn._solvers import conf_inf_solver, _process_args_kwargs_flatten, _return_x01_funs

class _truncnorm():
    def __init__(self, mu, sigma2, a, b) -> None:
        """
        Wrapper to transform the scipy.dists.truncnorm to the alpha/beta scale and handle parameter broadcasting
        
        Parameters
        ----------
        See _tnorm()

        Attributes
        ----------
        mu:                 Array of means
        sigma:              Array of standard deviations
        a:                  Array of lowerbounds
        b:                  Array of upperbounds
        alpha:              Array of z-score lowerbounds
        beta:               Array of z-score upperbounds
        dist:               The scipy truncnorm
        """
        # Input checks and broadcasting
        self.mu, sigma2, self.a, self.b = broastcast_max_shape(mu, sigma2, a, b)
        assert np.all(self.a < self.b), 'a needs to be < b'
        assert np.all(sigma2 > 0), 'sigma2 needs to be > 0'
        self.sigma = np.sqrt(sigma2)
        self.alpha = (self.a - self.mu) / self.sigma
        self.beta = (self.b - self.mu) / self.sigma
        # Initialize the distribution
        self.dist = truncnorm(loc=self.mu, scale=self.sigma, a=self.alpha, b=self.beta)


class _tnorm():
    def __init__(self, mu:float | np.ndarray | int, sigma2:float | np.ndarray | int, a:float | np.ndarray | int, b:float | np.ndarray | int, **kwargs) -> None:
        """
        Main model class for the truncated normal distribution

        Parameters
        ----------
        mu:                 Means of the unconditional normal (can be array)
        sigma2:             Variance of the truncated normal (can be array)
        a:                  Lower bounds of truncated normal (can be -infty)
        b:                  Upperbounds of truncated normal (can be +infty)
        kwargs:             Supported to absorb extranous parameters

        Attributes
        ----------
        _truncnorm.dist:    The scipy truncnorm distribution

        Methods
        -------
        Similar to scipy.dists

        cdf:        Calculates CDF
        pdf:        Calculates PDF
        ppf:        Calculates quantile
        rvs:        Generates random variables
        fit:        Tries to estimate mean
        """
        self._truncnorm = _truncnorm(mu, sigma2, a, b)


    def cdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).cdf()"""
        return self._truncnorm.dist.cdf(x)

    def ppf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).ppf()"""
        return self._truncnorm.dist.ppf(x)

    def pdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).pdf()"""
        return self._truncnorm.dist.pdf(x)

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
                fa = self._truncnorm.alpha
            if use_b:
                fb = self._truncnorm.beta
            if use_sigma:
                fsigma = self._truncnorm.sigma
            di = dict(zip(['fa','fb','fscale'],[fa,fb,fsigma]))
            di = {k:v for k,v in di.items() if v is not None}
            a_hat, b_hat, mu_hat, sigma_hat = truncnorm.fit(x, **di)
        else:
            # Solve for matrix case
            a_hat = self._truncnorm.mu * np.nan
            b_hat, mu_hat, sigma_hat = a_hat.copy(), a_hat.copy(), a_hat.copy()
            # Loop over all dimensions except the first
            for kk in np.ndindex(x.shape[1:]):
                x_kk = x[np.s_[:,] + kk]  # Extract as array
                if use_a:
                    fa = self._truncnorm.alpha[kk]
                if use_b:
                    fb = self._truncnorm.beta[kk]
                if use_sigma:
                    fsigma = self._truncnorm.sigma[kk]
                di = dict(zip(['fa','fb','fscale'],[fa,fb,fsigma]))
                di = {k:v for k,v in di.items() if v is not None}
                res_kk = truncnorm.fit(x_kk, **di)
                a_hat[kk], b_hat[kk], mu_hat[kk], sigma_hat[kk] = res_kk
        return a_hat, b_hat, mu_hat, sigma_hat


    def rvs(self, n:int, seed=None) -> np.ndarray:
        """Wrapper for scipy.stats.truncnorm(...).rvs()"""
        # When sampling it is [num_sample,*dims of parameters]
        samp_shape = (n,)
        if self._truncnorm.mu.shape != (1,):  # If everything is a float, no need for a second dimension
            samp_shape += self._truncnorm.mu.shape
        return self._truncnorm.dist.rvs(samp_shape, random_state=seed)

    @staticmethod
    def _dmu_dcdf(mu:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
        """
        For a given mean/point, determine the derivative of the CDF w.r.t. the location parameter. The argument construction follows what is expected by the conf_inf_solver class
        
        dF(mu)/du = {[(phi(alpha)-phi(xi))/sigma]*z + [(phi(alpha)-phi(beta))/sigma]*[Phi(xi)-Phi(alpha)] } /z^2
        z = Phi(beta) - Phi(alpha)

        Parameters
        ----------
        mu:                 Candidate array of means
        x:                  Realized points
        alpha:              Type-I error values
        args:               
        kwargs:             See accepted kwargs below

        **kwargs
        --------
        approx:             Should the _log_gauss_approx method be used (can help for tails of the distribution, but introduces some error)
        a_min:              Used for gradient clipping
        a_max:              Used for gradient clipping
        """
        # Process the args/kwargs
        flatten, kwargs = _process_args_kwargs_flatten(args, kwargs)
        approx, a_min, a_max = True, None, None
        if 'approx' in kwargs:
            if isinstance(kwargs['approx'], np.ndarray):
                approx = bool(kwargs['approx'][0])
            else:
                approx = kwargs['approx']
            assert isinstance(approx, bool), 'approx needs to be a bool'
        if 'a_min' in kwargs:
            a_min = kwargs['a_min']
        if 'a_max' in kwargs:
            a_max = kwargs['a_max']
        # Will error out if sigma/a/b not specified
        sigma2, a, b = kwargs['sigma2'], kwargs['a'], kwargs['b']
        sigma = np.sqrt(sigma2)
        # Type checks
        assert np.asarray(approx).dtype==bool, 'approx needs to be a boolean'
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
            dFdmu = grad_clip_abs(dFdmu, a_min, a_max)
        else:
            term1a = (norm.pdf(a_z)-norm.pdf(x_z))/sigma
            term1b = norm.cdf(b_z) - norm.cdf(a_z)
            term2a = norm.cdf(x_z) - norm.cdf(a_z)
            term2b = (norm.pdf(a_z)-norm.pdf(b_z))/sigma
            # quotient rule
            dFdmu = (term1a*term1b - term2a*term2b)/term1b**2  
        if flatten:
            dFdmu = np.diag(dFdmu.flatten())
        return dFdmu

    @staticmethod
    def _find_dist_kwargs_CI(**kwargs) -> tuple:
        """
        Looks for valid truncated normal distribution keywords that are needed for generating a CI

        Returns
        -------
        sigma2, a, b, a_min, a_max, kwargs
        """
        valid_kwargs = ['sigma2','a','b','a_min','a_max']
        # sigma2, a, b are required
        sigma2, a, b = kwargs['sigma2'], kwargs['a'], kwargs['b']
        # If a_{min,max} are not specified, assume they are the default value
        a_min, a_max = None, None
        
        if 'a_min' in kwargs:
            a_min = kwargs['a_min']
        if 'a_max' in kwargs:
            a_max = kwargs['a_max']
        # Remove only constructor arguments from kwargs
        kwargs = {k:v for k,v in kwargs.items() if k not in valid_kwargs}
        # Return constructor arguments and kwargs
        return sigma2, a, b, a_min, a_max, kwargs


    def conf_int(self, x:np.ndarray, alpha:float=0.05, approx:bool=True, **kwargs) -> np.ndarray:
        """
        Assume X ~ TN(mu, sigma, a, b), where sigma, a, & b are known. Then for any realized mu_hat (which can be estimated with self.fit()), we want to find the confidence interval for a series of points (x)

        Arguments
        ---------
        x:                      An array-like object of points that corresponds to dimensions of estimated means
        alpha:                  Type-1 error rate
        approx:                 Whether the Gaussian tail approximation should be used
        a_m{in/ax}:             Whether gradient clipping should be used
        fun_x01_type:           Whether a special x to initialization mapping should be applied (default='nudge'). See below. Will be ignored if fun_x{01} is provided
        kwargs:                 For other valid kwargs, see sntn._solvers._conf_int (e.g. a_min/a_max)
        """
        solver = conf_inf_solver(dist=_tnorm, param_theta='mu',dF_dtheta=self._dmu_dcdf, alpha=alpha)
        # Set up di_dist_args (these go into the tnorm class basically)
        if 'sigma2' not in kwargs:
            kwargs['sigma2'] = self._truncnorm.sigma**2
        if 'a' not in kwargs:
            kwargs['a'] = self._truncnorm.a
        if 'b' not in kwargs:
            kwargs['b'] = self._truncnorm.b
        sigma2, a, b, a_min, a_max, kwargs = self._find_dist_kwargs_CI(**kwargs)
        di_dist_args = {'sigma2':sigma2, 'a':a, 'b':b, 'a_min':a_min, 'a_max':a_max}
        di_dist_args['approx'] = approx
        # Run CI solver
        res = solver._conf_int(x=x, di_dist_args=di_dist_args, **kwargs)
        # Return matrix of values
        return res


