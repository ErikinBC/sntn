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
from sntn.utilities.utils import broastcast_max_shape, try2array, broadcast_to_k, reverse_broadcast_from_k

# Accepted CDF method
valid_cdf_approach = ['scipy', 'cox1', 'cox2', 'quad']



@staticmethod
def imills(a:np.ndarray) -> np.ndarray:
    """Returns the inverse mills ratio"""
    return norm.pdf(a)/norm.cdf(-a)

@staticmethod
def sheppard(theta:np.ndarray, h:np.ndarray, k:np.ndarray) -> np.ndarray:
    """Returns the function value for doing a BVN integral"""
    return (1/(2*np.pi))*np.exp(-0.5*(h**2+k**2-2*h*k*np.cos(theta))/(np.sin(theta)**2))


class _bvn():
    def __init__(self, mu1:float or np.ndarray, sigma21:float or np.ndarray, mu2:float or np.ndarray, sigma22:float or np.ndarray, rho:float or np.ndarray, cdf_approach:str='scipy') -> None:
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


    def cdf(self, x:np.ndarray) -> np.ndarray:
        """
        Calculate the CDF of a pair of points. If x is a 1-d array, then it needs to match the dimensions of the constructor parameters
        """
        x = try2array(x)
        if nz_shape > 1:
            1
        else:
            assert self.param_shape == z.shape, 'If x is a vector (or float), then it must match the dimensions of the input. For example, if mu1=[1,2,3], then '        


    def rvs(self, ndraw:int, seed=None) -> np.ndarray:
        """
        Get ndraw samples from the underlying distributions
        
        Parameters
        ----------
        ndraw:          Number of samples to simulate
        seed:           Seed to pass onto np.random.seed

        Returns
        -------
        An (ndraw,2,k) array of simulated values
        """
        np.random.seed(seed)
        x = np.random.randn(self.k, 2, ndraw)
        z = np.einsum('ijk,ikl->ijl', self.A.reshape([self.k,2,2]).transpose(0,2,1), x).transpose(2,1,0)
        z += np.expand_dims(np.c_[self.mu1, self.mu2].T,0)
        # Return to original shape
        z = reverse_broadcast_from_k(z, self.param_shape)
        return z


    # h, k = -2, -np.infty
    def orthant(self, h, k, method='scipy'):
        # P(X1 >= h, X2 >=k)
        assert method in ['scipy','cox','sheppard']
        if isinstance(h,int) or isinstance(h, float):
            h, k = np.array([h]), np.array([k])
        else:
            assert isinstance(h,np.ndarray) and isinstance(k,np.ndarray)
        assert len(h) == len(k)
        # assert np.all(h >= 0) and np.all(k >= 0)
        # Calculate the number of standard deviations away it is        
        Y = (np.c_[h, k] - self.mu)/np.sqrt(self.sigma)
        Y1, Y2 = Y[:,0], Y[:,1]
        
        # (i) scipy: L(h, k)=1-(F1(h)+F2(k))+F12(h, k)
        if method == 'scipy':
            sp_bvn = MVN([0, 0],[[1,self.rho],[self.rho,1]])
            pval = 1+sp_bvn.cdf(Y)-(norm.cdf(Y1)+norm.cdf(Y2))
            return pval 

        # A Simple Approximation for Bivariate and Trivariate Normal Integrals
        if method == 'cox':
            mu_a = self.imills(Y1)
            root = np.sqrt(1-self.rho**2)
            xi = (self.rho * mu_a - Y2) / root
            pval = norm.cdf(-Y1) * norm.cdf(xi)
            return pval

        if method == 'sheppard':
            pval = np.array([quad(self.sheppard, np.arccos(self.rho), np.pi, args=(y1,y2))[0] for y1, y2 in zip(Y1,Y2)])
            return pval
