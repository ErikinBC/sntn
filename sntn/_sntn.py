"""
Fully specified SNTN distribution
"""

# External
import numpy as np
from scipy.stats import truncnorm, norm
from inspect import getfullargspec
# Internal
# from sntn.utilities.grad import _log_gauss_approx, _log_diff
from sntn.utilities.utils import broastcast_max_shape
from sntn._solvers import conf_inf_solver, _process_args_kwargs_flatten


class _nts():
    def __init__(self, mu1:float or np.ndarray or None, tau1:float or np.ndarray, mu2:float or np.ndarray or None, tau2:float or np.ndarray, a:float or np.ndarray, b:float or np.ndarray, c1:float or np.ndarray=1, c2:float or np.ndarray=1, fix_mu:bool=False) -> None:
        """
        The "normal and truncated sum": workhorse class for the sum of a normal and truncated normal. Carries out standard inferece using scipy.dist syntax with added conf_int method

        W = c1*Z1 + c2*Z2,  Z1 ~ N(mu1, tau1^2), Z2 ~ TN(mu2, tau2^2, a, b)
        W ~ NTS(theta(mu(c)), Sigma(tau(c)), a, b)
        mu(c) =             [c1*mu1, c2*mu2]
        tau(c) =            [c1**2 * tau1, c2**2 * tau2]
        theta(mu(c)) =      [sum(mu), mu[1]]
        sigma2 =            [sum(tau), tau[1]]
        sigma =             sqrt(sigma2)
        rho =               sigma[1] / sigma[0]
        Sigma(tau(c)) =     [ [sigma2[0], sigma2[1]], [sigma2[1], sigma2[1]]]
        
        
        Parameters
        ----------
        mu1:             Mean of the unconditional gaussian (can be None is fix_mu=True)
        tau1:            The variance of the unconditional gaussian (must be strictly positive)
        mu2:             The mean of the truncated Gaussian (can be None is fix_mu=True)
        tau2:            The variance of the unconditional gaussian (must be strictly positive)
        a:               The lowerbound of the truncated Gaussian (must be smaller than b)
        b:               The upperbound of the truncated Gaussian (must be larger than a)
        c1:              A weighting constant for Z1 (must be >0, default=1)
        c2:              A weighting constant for Z2 (must be >0, default=1)
        fix_mu:          Whether mu1=mu2=mu, will need mu1 OR mu2 to be None, but not both (default=False)

        Attributes
        ----------
        k:              Number of NTS distributiosn (>1 when inputs are not floats/ints)
        theta:          A (k,2) array to parameterize NTS
        Sigma:          A (k,4) array with covariance matrix
        a:              A (k,) array of lower bounds
        b:              A (k,) array of upper bounds

        Methods
        -------
        cdf:            Cumulative distribution function
        pdf:            Density function
        ppf:            Quantile function

        """
        # Input checks and broadcasting
        if fix_mu:
            mu1, mu2 = self._mus_are_equal(mu1, mu2)
        mu1, mu2, tau1, tau2, a, b, c1, c2 = broastcast_max_shape(mu1, mu2, tau1, tau2, a, b, c1, c2)
        assert np.all(tau1 > 0), 'tau1 needs to be strictly greater than zero'
        assert np.all(tau2 > 0), 'tau1 needs to be strictly greater than zero'
        assert np.all(b > a), 'b needs to be greated than a'
        assert np.all(c1 > 0), 'c1 needs to be strictly greater than zero'
        assert np.all(c2 > 0), 'c2 needs to be strictly greater than zero'
        # Capture the original shape for later transformations
        self.param_shape = mu1.shape
        # Flatten parameters
        mu1, mu2, tau1, tau2, a, b, c1, c2 = [x.flatten() for x in [mu1, mu2, tau1, tau2, a, b, c1, c2]]
        # Create attributes
        self.k = len(mu1)
        self.theta = np.c_[c1*mu1 + c2*mu2, c2*mu2]
        tau = np.c_[c1**2 * tau1, c2**2 * tau2]
        sigma2 = np.c_[np.sum(tau, axis=1, keepdims=True), tau[:,1]]
        sigma = np.sqrt(sigma2)
        rho = sigma[:,1]/ sigma[:,0]
        self.Sigma = np.c_[sigma2[:,0], sigma2[:,1], sigma2[:,1], sigma2[:,1]]
        self.a = a
        self.b = a


    @staticmethod
    def _mus_are_equal(mu1, mu2) -> tuple:
        """When we desire to fix mu1/mu2 to the same value, will return the value of both when one of them is None"""
        if mu1 is None:
            assert mu2 is not None, 'if mu1 is None, mu2 cannot be'
            return mu2, mu2
        elif mu2 is None:
            assert mu1 is not None, 'if mu1 is None, mu2 cannot be'
            return mu1, mu1
        else:
            assert np.all(mu1 == mu2), 'if mu is fixed, mu1 != mu2'
            return mu1, mu2


    def cdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Returns the cumulative distribution function"""
        return None


    @staticmethod
    def _dmu_dcdf(mu:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
        """Return the derivative of...."""
        return None


    def _find_dist_kwargs_CI(**kwargs) -> tuple:
        """
        Looks for valid truncated normal distribution keywords that are needed for generating a CI

        Returns
        -------
        {mu1,mu2}, tau1, tau2, c1, c2, fix_mu, kwargs
        """
        valid_kwargs = ['mu1', 'mu2', 'tau1', 'tau2', 'c1', 'c2', 'fix_mu']
        # tau1/tau2 are always required
        tau1, tau2 = kwargs['tau1'], kwargs['tau2']
        # If c1/c2/fix_mu are not specified, assume they are the default values
        if 'c1' not in kwargs:
            c1 = 1
        if 'c2' not in kwargs:
            c2 = 1
        if 'fix_mu' not in kwargs:
            fix_mu = False
        # Remove only constructor arguments from kwargs
        kwargs = {k:v for k,v in kwargs.items() if k not in valid_kwargs}
        # We should only see mu1 or mu2
        has_mu1 = 'mu1' in kwargs
        has_mu2 = 'mu2' in kwargs
        assert not has_mu1 and has_mu2, 'We can only do inference on mu1 or mu2'
        assert has_mu1 or has_mu2, 'mu1 OR mu2 needs to be specified'
        if has_mu1:
            mu = kwargs['mu1']
        if has_mu2:
            mu = kwargs['mu2']
        # Remove valid_kwargs from kwargs
        kwargs = {k:v for k,v in kwargs.items() if k not in valid_kwargs}
        # Return constructor arguments and kwargs
        return mu, tau1, tau2, c1, c2, fix_mu, kwargs


    def conf_int(self, x:np.ndarray, alpha:float=0.05, param_fixed:str='mu', **kwargs) -> np.ndarray:
        """
        Assume W ~ NTS()...

        Confidence intervals for the NTS distribution are slightly different than other distribution because theta is a function of mu1/mu2. Instead we can do CIs for mu1 or mu2, or mu=mu1=mu2 (see fix_mu in the constructor)

        Arguments
        ---------
        x:                      An array-like object of points that corresponds to dimensions of estimated means
        alpha:                  Type-1 error rate
        param_fixed:            Which parameter are we doing inference on ('mu'==fix mu1==mu2, 'mu1', 'mu2')? 
        kwargs:                 For other valid kwargs, see sntn._solvers._conf_int (e.g. a_min/a_max)
        """
        # Process the key-word arguments and extract NTS parameters
        mu, tau1, tau2, c1, c2, fix_mu, kwargs = self._find_dist_kwargs_CI(**kwargs)
        # Storage for the named parameters what will go into class initialization (excluded param_fixed)
        di_dist_args = {}
        # param_fixed must be either mu1, mu2, or mu (mu1==mu2)
        valid_fixed = ['mu1', 'mu2', 'mu']
        assert param_fixed in valid_fixed, f'param_fixed must be one of: {param_fixed}'
        if param_fixed == 'mu':
            param_fixed = 'mu1'  # Use mu1 for convenience
            assert fix_mu, 'if param_fixed="mu" then fix_mu==True'
            di_dist_args['mu2'] = None  # One parameter must be None with fix_mu=True, and since mu1 will be assigned every time, this is necessary
        elif param_fixed == 'mu1':
            param_fixed = 'mu1'
            di_dist_args['mu2'] = mu
        else:
            param_fixed = 'mu2'
            di_dist_args['mu1'] = mu
        # Set up solver
        solver = conf_inf_solver(dist=_nts, param_theta=param_fixed, dF_dtheta=self._dmu_dcdf, alpha=alpha)
        # Assign the remainder of the parameter
        di_dist_args = {**di_dist_args, **{'tau1':tau1, 'tau2':tau2, 'c1':c1, 'c2':c2, 'fix_mu':fix_mu}}
        # Run CI solver
        res = solver._conf_int(x=x, di_dist_args=di_dist_args, **kwargs)
        # Return matrix of values
        return res
