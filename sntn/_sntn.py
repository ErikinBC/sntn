"""
Fully specified SNTN distribution
"""

# External
import numpy as np
from scipy.stats import truncnorm, norm
from inspect import getfullargspec
# Internal
from sntn.utilities.utils import broastcast_max_shape
from sntn._solvers import conf_inf_solver, _process_args_kwargs_flatten


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
        alpha:          A (k,) array of how many SDs the lowerbound is from the mean
        beta:           A (k,) array of how many SDs the upperbound is from the mean
        Z:              A (k,) array of 
        dist_Z1:        A scipy.stats.norm distribution
        dist_Z2:        A scipy.stats.truncnorm distribution

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
        assert np.all(tau2 > 0), 'tau2 needs to be strictly greater than zero'
        assert np.all(b > a), 'b needs to be greated than a'
        assert np.all(c1 > 0), 'c1 needs to be strictly greater than zero'
        assert np.all(c2 > 0), 'c2 needs to be strictly greater than zero'
        # Capture the original shape for later transformations
        self.param_shape = mu1.shape
        # Flatten parameters
        mu1, mu2, tau1, tau2, a, b, c1, c2 = [x.flatten() for x in [mu1, mu2, tau1, tau2, a, b, c1, c2]]
        # Create attributes
        self.k = len(mu1)
        self.theta1 = c1*mu1 + c2*mu2
        self.theta2 = c2*mu2
        self.sigma1 = c1**2 * tau1 + c2**2 * tau2
        self.sigma2 = c2**2 * tau2
        self.rho = self.sigma2 / self.sigma1
        # Calculate the truncated normal terms
        self.alpha = (a - mu2) / np.sqrt(tau2)
        self.beta = (b - mu2) / np.sqrt(tau2)
        self.Z = norm.cdf(self.beta) - norm.cdf(self.alpha)
        # Use the scipy classes to create separate distributions
        self.dist_Z1 = norm(loc=c1*mu1, scale=c1*np.sqrt(tau1))
        self.dist_Z2 = truncnorm(loc=c2*mu2, scale=c2*np.sqrt(tau2), a=self.alpha, b=self.beta)


    def mean(self) -> np.ndarray:
        """Calculate the mean of the NTS distribution"""
        mu = self.dist_Z1.mean() + self.dist_Z2.mean()
        mu = mu.reshape(self.param_shape)
        return mu


    def cdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Returns the cumulative distribution function"""
        return None


    def pdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the marginal density of the NTS distribution at some point x"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        assert x.shape[-1] == self.k, f'Last dim of x needs to match {self.k}'
        term1 = self.sigma1 * self.Z
        m1 = (x - self.theta1) / self.sigma1
        term2 = (self.beta-self.rho*m1) / np.sqrt(1-self.rho**2)
        term3 = (self.alpha-self.rho*m1) / np.sqrt(1-self.rho**2)
        f = norm.pdf(m1)*(norm.cdf(term2) - norm.cdf(term3)) / term1
        f = self.to_original_shape(f)
        return f


    def to_original_shape(self, z:np.ndarray) -> np.ndarray:
        """Returns to the original shape"""
        nz_shape = len(z.shape)
        if nz_shape > 1:
            # Assume its from rvs
            return z.reshape((z.shape[0],)+self.param_shape)
        else:
            assert self.param_shape == z.shape, 'If not rvs, then shapes need to match'
            return z.reshape(self.param_shape)
        

    def rvs(self, ndraw:int, seed=None) -> np.ndarray:
        """Generate n samples from the distribution"""
        z1 = self.dist_Z1.rvs([ndraw,self.k], random_state=seed)
        z2 = self.dist_Z2.rvs([ndraw, self.k], random_state=seed)
        w = z1 + z2
        w = self.to_original_shape(w)
        return w


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
