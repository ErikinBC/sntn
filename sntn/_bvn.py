"""
Main bivariate normal class
"""

# External
import numpy as np
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as MVN
from sklearn.linear_model import LinearRegression
# Internal
from sntn._cdf_bvn._approx import _bvn_cox
from sntn._cdf_bvn._brute import _bvn_scipy
from sntn.utilities.utils import broastcast_max_shape, try2array, broadcast_to_k, reverse_broadcast_from_k


# Accepted CDF method
valid_cdf_approach = ['scipy', 'cox1', 'cox2']  #, 'quad'


class _bvn():
    def __init__(self, mu1:float or np.ndarray, sigma21:float or np.ndarray, mu2:float or np.ndarray, sigma22:float or np.ndarray, rho:float or np.ndarray, cdf_approach:str='scipy', **kwargs) -> None:
        """
        Main workhorse class for a bivariate normal distribution:

        X1 ~ N(mu1, sigma21), X2 ~ N(mu2, sigma22)
        corr(X1, X2) = rho
        [X1; X2] ~ BVN( [mu1, mu2], [[sigma21, rho], [rho, sigma22]] )
        
        Parameters
        ----------
        mu1:                    The mean of the first Gaussian
        sigma21:                The variance of the first Gaussian
        mu2:                    The mean of the second Gaussian
        sigma22:                The variance of the second Gaussian
        rho:                    The correlation b/w X1, X2
        cdf_approach:           Which approach should be used to calculate CDF? (default='scipy')
        kwargs:                 Any other keywords to pass into a cdf_approach (e.g. nsim=1000 for cox1)

        CDF approaches
        ----------
        scipy:                  Uses a grid approach for each BVN pair
        cox1:                   The simplest approach to estimate CDF (first-order)
        cox2:                   Includes second-order approximation terms
        quad:                   A quadrature approach to estimate the integral


        Methods
        -------

        """
        # Process inputs
        mu1, sigma21, mu2, sigma22, rho = broastcast_max_shape(mu1, sigma21, mu2, sigma22, rho)
        # Input checks
        assert np.all((rho >= -1) & (rho <= +1)), 'rho needs to be b/w [-1,+1]'
        assert np.all(sigma21 > 0), 'sigma21 needs to be > 0'
        assert np.all(sigma22 > 0), 'sigma22 needs to be > 0'
        assert isinstance(cdf_approach, str), 'cdf_approach needs to be a string'
        assert cdf_approach in valid_cdf_approach, f'cdf approach must be one of: {valid_cdf_approach}'
        # Capture the original shape for later transformations
        self.param_shape = mu1.shape
        # Prepare cdf method (important to assign before we flatten as well)
        self.cdf_approach = cdf_approach
        if self.cdf_approach == 'cox1':
            self.cdf_method = _bvn_cox(mu1, mu2, sigma21, sigma22, rho, monte_carlo=True, **kwargs)
        if self.cdf_approach == 'cox2':
            self.cdf_method = _bvn_cox(mu1, mu2, sigma21, sigma22, rho, monte_carlo=False, **kwargs)
        if self.cdf_approach == 'scipy':
            self.cdf_method = _bvn_scipy(mu1, mu2, sigma21, sigma22, rho)
        
        # Flatten parameters
        mu1, sigma21, mu2, sigma22, rho = [x.flatten() for x in [mu1, sigma21, mu2, sigma22, rho]]
        # Assign attributes
        self.k = len(mu1)
        self.mu1 = mu1
        self.sigma21 = sigma21 
        self.mu2 = mu2
        self.sigma22 = sigma22 
        self.rho = rho
        self.sigma1 = np.sqrt(self.sigma21)
        self.sigma2 = np.sqrt(self.sigma22)
        
        # Create the Sigma covariance matrix, which is a (k,4 matrix) which will be looped through for cholesky decomposition as well
        self.Sigma = np.stack([self.sigma21, self.sigma1*self.sigma2*self.rho, self.sigma1*self.sigma2*self.rho, self.sigma22],axis=1)
        # Cholesky decomp used for for drawing data        
        self.A = np.zeros(self.Sigma.shape)
        for i in range(self.k):
            self.A[i] = cholesky(self.Sigma[i].reshape(2,2)).flatten()


    def cdf(self, x:np.ndarray or None=None, x1:np.ndarray or None=None, x2:np.ndarray or None=None) -> np.ndarray:
        """
        Calculates the CDF for an array with two dimensions (i.e. bivariate normal)

        Parameters
        ----------
        x (np.ndarray):
            A (n1,..,nj,d1,..,dk,2) array
        x1 (np.ndarray):
            If x is not specified...
        x2 (np.ndarray):
            Second coordinate if x not specified

        Returns
        -------
        An (n1,..,nj,d1,..,dk) array of CDF values
        """
        if x is None:
            assert x1 is not None and x2 is not None, 'if x is not specified, then x1/x2 need to be'
            x1, x2 = np.broadcast_arrays(x1, x2)
        else:
            x = np.asarray(x)
            assert x.shape[-1] == 2, 'if x is given (rather than x1/x2), last dimension needs to be of size 2'
            # Take out x1/x2
            x1, x2 = np.take(x, 0, -1), np.take(x, 1, -1)
        pval = self.cdf_method.cdf(x1, x2)
        return pval


    def rvs(self, ndraw:int, seed=None) -> np.ndarray:
        """
        Get ndraw samples from the underlying distributions
        
        Parameters
        ----------
        ndraw:          Number of samples to simulate
        seed:           Seed to pass onto np.random.seed

        Returns
        -------
        An (ndraw,d1,...dk,2) array of simulated values
        """
        np.random.seed(seed)
        x = np.random.randn(self.k, 2, ndraw)
        z = np.einsum('ijk,ikl->ijl', self.A.reshape([self.k,2,2]).transpose(0,2,1), x).transpose(2,1,0)
        z += np.expand_dims(np.c_[self.mu1, self.mu2].T,0)
        # Return to original shape
        z = reverse_broadcast_from_k(z, self.param_shape)
        # Put second dimension last
        z = z.transpose([0]+list(range(2,len(z.shape))) + [1])
        return z
