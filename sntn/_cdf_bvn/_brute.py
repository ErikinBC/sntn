"""
Brute force approaches for BVN integral
"""

# External
import numpy as np
from scipy.special import owens_t
from scipy.integrate import fixed_quad
from scipy.stats import multivariate_normal as mvn
# Internal
from sntn._cdf_bvn._utils import _bvn_base, mvn_pivot, Phi, orthant_to_cdf

# Valid quadrature methods
valid_quad_approach = ['owen', 'drezner1', 'drezner2']


#####################
# --- (1) SCIPY --- #


class _bvn_scipy(_bvn_base):
    def __init__(self, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray):
        super().__init__(mu1, mu2, sigma21, sigma22, rho)


    def cdf(self, x1:np.ndarray, x2:np.ndarray) -> np.ndarray:
        # Converts to normalized units
        h, k, rho = mvn_pivot(x1, x2, self.mu1, self.mu2, self.sigma21, self.sigma22, self.rho)
        h_shape = h.shape
        n_h_shape = len(h_shape)
        # Initialize holders
        pval = np.zeros(h.shape)
        Sigma = np.array([[1,0],[0,1]], dtype=h.dtype)
        # If h.shape == mu1.shape, then broadcasting did not affect values, and we assume we need to apply a 1:1 mapping, however if h.shape > mu1.shape, when clearly there are redundant parameters that can be called more efficiently with cdf
        if (self.n_param_shape == 0) or (self.n_param_shape==1 and sum(self.param_shape)==1):
            # parameters are floats so, cdf can be vectorized
            Sigma[[0,1],[1,0]] = self.rho
            data = np.stack([h,k])
            assert data.shape[0] == 2, 'expected first dimension to be of size 2'
            n = int(np.prod(data.shape[1:]))
            pval = mvn(cov=Sigma).cdf(data.reshape([2, n]).T)
            pval = pval.reshape(h.shape)
        elif self.n_param_shape == n_h_shape:
            # Assumes we need a 1:1 mapping
            for ii in np.ndindex(h.shape):
                Sigma[[0,1],[1,0]] = rho[ii]
                pval[ii] = mvn(cov=Sigma).cdf(np.r_[h[ii], k[ii]])
        else:
            # We should be able to recycle coordinates
            assert self.n_param_shape < n_h_shape
            # Determine which dimensions are "recycles"
            n_dim_recycle = n_h_shape - self.n_param_shape
            for ii in np.ndindex(h_shape[:n_dim_recycle]):
                # Make a recursive call...
                pval[*ii,...] = self.cdf(x1[*ii,...], x2[*ii,...])
        return pval




##########################
# --- (2) QUADRATURE --- #

class _bvn_quad(_bvn_base):
    def __init__(self, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray, quad_approach:str='owen'):
        # Construct base class
        super().__init__(mu1, mu2, sigma21, sigma22, rho)
        # Input checks
        assert quad_approach in valid_quad_approach, f'quad_approach must be one of {valid_quad_approach}'
        if quad_approach == 'owen':
            self.fun_cdf = self._cdf_owen
        if quad_approach == 'drezner1':
            self.fun_cdf = self._cdf_drezner1
        if quad_approach == 'drezner2':
            self.fun_cdf = self._cdf_drezner2


    def cdf(self, x1:np.ndarray, x2:np.ndarray, **kwargs) -> np.ndarray:
        """
        Calls the quad_approach to calculate the CDF
        
        Parameters
        ----------
        x{12}:                      A (d1,d2,..dk) array of the first and second data coordinate
        
        """
        # Process x1/x2
        h, k, rho = mvn_pivot(x1, x2, self.mu1, self.mu2, self.sigma21, self.sigma22, self.rho)
        res = self.fun_cdf(h, k, rho, **kwargs)
        return res


    @staticmethod
    def _cdf_owen(h:np.ndarray, k:np.ndarray, rho:np.ndarray) -> np.ndarray:
        """Owen's (1956) method, where the 'intergral' is the owens_t function"""
        # Calculate delta_hk (either zero or one), formally: np.where(h*k>=0, 0, np.where(h + k >= 0, 0, 1))
        delta_hk = np.where(h*k>=0, 0, 1)  
        # Calculate constants for Owen's T
        den_rho = np.sqrt(1 - rho**2)
        q1 = (k / h - rho) / den_rho
        q2 = (h / k - rho) / den_rho
        # The "owens_t" function is fully vectorized and performs the integral under the hood
        t1 = owens_t(h, q1)
        t2 = owens_t(k, q2)
        cdf = 0.5 * (Phi(h) + Phi(k) - delta_hk) - t1 - t2 
        return cdf

    @staticmethod
    def _drezner1(r:np.ndarray, h:np.ndarray, k:np.ndarray) -> np.ndarray:
        """
        Drezner and Wesolowsky's (1990) approach to converting a two-variable BVN CDF to a single parameter

        \Phi(-h)\Phi(-k) + \frac{1}{2\pi} \int_0^\rho \frac{1}{1-r^2} \exp\Big[ -\frac{h^2 + k^2 - 2hkr}{2 (1-r)^2}  \Big] dr
        """
        term1 = h**2 + k**2 - 2*h*k*r
        term2 = 2 * (1 - r**2)
        term3 = 1 / np.sqrt(1 - r**2)
        const = 1 / (2*np.pi)
        f = term3*np.exp(-term1 / term2) * const
        return f


    def _cdf_drezner1(self, h:np.ndarray, k:np.ndarray, rho:np.ndarray, **kwargs) -> np.ndarray:
        """Runs vectorized quadrature to solve the Drezner 1 approach"""
        orthant = np.zeros(h.shape)
        for ii in np.ndindex(h.shape):
            h_ii, k_ii = h[ii], k[ii]
            res_ii = fixed_quad(func=self._drezner1, a=0, b=rho[ii], args=(h_ii, k_ii), **kwargs)[0]
            res_ii += Phi(-h_ii) * Phi(-k_ii)
            orthant[ii] = res_ii
        cdf = orthant_to_cdf(orthant, h, k)
        return cdf


    def _cdf_drezner2(self, h:np.ndarray, k:np.ndarray, rho:np.ndarray, **kwargs) -> np.ndarray:
        """Runs vectorized quadrature to solve the Drezner 2 approach"""
        orthant = np.zeros(h.shape)
        for ii in np.ndindex(h.shape):
            h_ii, k_ii = h[ii], k[ii]
            res_ii = fixed_quad(func=self._drezner2, a=0, b=np.arcsin(rho[ii]), args=(h_ii, k_ii), **kwargs)[0]
            res_ii += Phi(-h_ii) * Phi(-k_ii)
            orthant[ii] = res_ii
        cdf = orthant_to_cdf(orthant, h, k)
        return cdf


    @staticmethod
    def _drezner2(theta:np.ndarray, h:np.ndarray, k:np.ndarray) -> np.ndarray:
        """
        Drezner and Wesolowsky's (1990) approach to converting a two-variable BVN CDF to a single parameter

        \Phi(-h)\Phi(-k) + \frac{1}{2\pi} \int_0^{\arcsin(\rho)} \exp\Big[ -\frac{h^2 + k^2 - 2hk\sin(\theta)}{2 \cos^2(\theta)}  \Big] d\theta
        """
        term1 = h**2 + k**2 - 2*h*k*np.sin(theta)
        term2 = 2 * np.cos(theta)**2
        const = 1/(2*np.pi)
        f = const*np.exp(-term1 / term2)
        return f