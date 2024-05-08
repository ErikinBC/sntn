"""
Main bivariate normal class
"""

# External
import numpy as np
from scipy.linalg import cholesky
# Internal
from sntn.utilities.utils import process_x_x1_x2
from sntn._cdf_bvn._approx import bvn_cox, valid_cox_approach
from sntn._cdf_bvn._brute import bvn_scipy, valid_quad_approach, _bvn_quad
from sntn.utilities.utils import broastcast_max_shape, reverse_broadcast_from_k

# Accepted CDF methods
valid_cdf_approach = ['scipy'] + valid_cox_approach + valid_quad_approach


class _bvn():
    def __init__(self, mu1:float | np.ndarray, sigma21:float | np.ndarray, mu2:float | np.ndarray, sigma22:float | np.ndarray, rho:float | np.ndarray, cdf_approach:str='owen', **kwargs) -> None:
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
        cox1:                   Uses a monte carlo approach to estimate CDF (specify "nsim" as a kwarg)
        cox2:                   Uses an approximation of the expectation term
        owen:                   Uses Owen's T function
        drezner1:               Uses fixed_quad to estimate CDF (in "rho" space)
        drezner2:               Uses fixed_quad to estimate CDF (in "cos" space)

        Methods
        -------
        cdf:                    Calculates P(X1 <= x1, X2 <= x2)
        rvs:                    Generates psuedo-random samples via cholesky decomposition
        """
        # Process inputs
        if isinstance(cdf_approach, np.ndarray):
            cdf_approach = str(cdf_approach.flat[0])  # Assume it was broadcasted
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
            self.cdf_method = bvn_cox(mu1, mu2, sigma21, sigma22, rho, monte_carlo=True, **kwargs)
        if self.cdf_approach == 'cox2':
            self.cdf_method = bvn_cox(mu1, mu2, sigma21, sigma22, rho, monte_carlo=False, **kwargs)
        if self.cdf_approach == 'scipy':
            self.cdf_method = bvn_scipy(mu1, mu2, sigma21, sigma22, rho)
        if self.cdf_approach == 'owen':
            self.cdf_method = _bvn_quad(mu1, mu2, sigma21, sigma22, rho, 'owen')
        if self.cdf_approach == 'drezner1':
            self.cdf_method = _bvn_quad(mu1, mu2, sigma21, sigma22, rho, 'drezner1')
        if self.cdf_approach == 'drezner2':
            self.cdf_method = _bvn_quad(mu1, mu2, sigma21, sigma22, rho, 'drezner2')
        
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


    def cdf(self, x:np.ndarray | None=None, x1:np.ndarray | None=None, x2:np.ndarray | None=None, **kwargs) -> np.ndarray:
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
        x1, x2 = process_x_x1_x2(x, x1, x2)
        pval = self.cdf_method.cdf(x1, x2, **kwargs)
        # pval = np.clip(pval, 0.0, 1.0)
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
        z = np.einsum('ijk,ikl->ijl', self.A.reshape([self.k,2,2]).transpose(0,2,1), x)
        z = z.transpose(2,0,1)
        z += np.c_[self.mu1, self.mu2]
        # Return to original shape
        z = reverse_broadcast_from_k(z, self.param_shape, suffix_shape=(2,))
        # # Put second dimension last
        # z = z.transpose([0]+list(range(2,len(z.shape))) + [1])
        return z
