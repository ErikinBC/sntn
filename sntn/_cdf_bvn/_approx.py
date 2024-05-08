"""
Contains the Cox method to approximate Bivariate CDF
"""

# External
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import owens_t
# Internal
from sntn.utilities.utils import try2array
from sntn._cdf_bvn._utils import _bvn_base, Phi, phi, mvn_pivot, orthant_to_cdf, imills

# For the _bvn class
valid_cox_approach = ['cox1', 'cox2']


class bvn_cox(_bvn_base):
    def __init__(self, mu1:np.ndarray, mu2:np.ndarray, sigma21:np.ndarray, sigma22:np.ndarray, rho:np.ndarray, rho_thresh:float=0.9, rho_max_w:float=0.5, monte_carlo:bool=False, nsim:int=1000, seed:int or None=None):
        super().__init__(mu1, mu2, sigma21, sigma22, rho)
        """Method for obtaining the CDF of a bivariate normal using Cox's 1991 approximation method
        
        Parameters
        ----------
        mu{12}:                     A (d1,d2,..dk) array of means
        sigma2{12}:                 A (d1,d2,..dk) array of variances
        rho:                        A (d1,d2,..dk) array of correlation coefficients
        rho_thresh:                 How high should abs(rho) be before averaging with Taylor approximation (default=0.9)
        rho_max_w:                  Maximum weight to apply to the Taylor approx to update solution
        monte_carlo:                Should random data be drawn to estimate P(X2 > k | X1 > h)? default=False <--> we use ~E[ Phi((rho*X1 - k)/sqrt(1-rho^2)) | X1 > h] 
        nsim:                       If MC used, how many simulated points to draw?
        seed:                       If MC used, how the draws should be seeded
        """
        # Input checks
        assert isinstance(nsim, int) and (nsim > 0), 'nsim needs to be a strictly positive int'
        assert isinstance(monte_carlo, bool), 'monte_carlo needs to be a bool'
        assert (rho_thresh >= 0) and (rho_thresh <= 1), 'rho_thresh needs to be b/w 0-1'
        assert (rho_max_w >= 0) and (rho_max_w <= 1), 'rho_max_w needs to be b/w [0,1]'
        # Assign for later
        self.rho_thresh = rho_thresh
        self.rho_max_w = rho_max_w
        self.monte_carlo = monte_carlo
        self.nsim = nsim
        self.seed = seed


    def cdf(self, x1:np.ndarray, x2:np.ndarray, monte_carlo=None, return_orthant:bool=False) -> np.ndarray:
        """
        Will default ot call cdf_cox1 (monte carlo) or cdf_cox2 (approximation) depending on choice at construction (but can be overwritten with function call)
        
        Parameters
        ----------
        x{12}:                      A (d1,d2,..dk) array of the first and second data coordinate
        
        """
        # Assign default if not provided
        if monte_carlo is None:
            monte_carlo = self.monte_carlo
        assert isinstance(monte_carlo, bool), 'if monte_carlo overrides, needs to be bool'
        fun_cdf = self.cdf_cox2
        if monte_carlo:
            fun_cdf = self.cdf_cox1
        # Process x1/x2
        h, k, rho = mvn_pivot(x1, x2, self.mu1, self.mu2, self.sigma21, self.sigma22, self.rho)
        res = fun_cdf(h, k, rho)
        # Ensure positivity (negativity can happen with Taylor expansion....)
        if not monte_carlo:
            res = np.clip(res, 0, 1)
        if return_orthant:
            res = cdf_to_orthant(res, h, k)
        return res

    def cdf_cox1(self, h, k, rho) -> np.ndarray:
        """
        Implements a Monte Carlo approach of estimating the orthant probability:

        L(h, k, rho) = P(X1 > h)*P(X2 > k | X1 > h)
                    = Phi(-h) * E[ Phi((rho*X1 - k)/sqrt(1-rho^2)) | X1 > h] 
        
        Where X1 > h is simply a truncated Gaussain: TN(0,1,h,infty)

        Parameters
        ----------
        h:             A (d1,d2,...,dk) array of normalized X1's
        k:             A (d1,d2,...,dk) array of normalized X1's
        rho:           A (d1,d2,...,dk) array of correlation coefficients
        nsim:          How many draws from the truncated normal to take (default=1000)
        seed:          Will be passed into rvs method of truncnorm (default=None)
        """
        # Draw data to approximate E[...] term
        dist = truncnorm(loc=0, scale=1, a=h, b=np.infty)
        rvs = dist.rvs((self.nsim,)+h.shape, self.seed)
        orthant = Phi(-h) * np.mean(Phi((rho*rvs - k)/np.sqrt(1-rho**2)),0)
        # Convert to CDF
        cdf = orthant_to_cdf(orthant, h, k)
        # In case orthant probabilities map to a negative value
        cdf = np.clip(cdf, 1/(self.nsim + 1), 1)
        return cdf


    def cdf_cox2(self, h, k, rho) -> np.ndarray:
        """
        Approximates the orthant probability using approach suggested by Cox (1991) and then converts to CDF. Uses a blended averages of equation 3/4, where weight is 50/50 as rho approaches 1

        Parameters
        ----------
        ...                     See cdf_cox1
        rho_thresh: 
        """
        # Get baseline results
        orthant = self._orthant_cox_rho(h, k, rho, first_order=False)
        # If there are any rho's above 90%, we will adjust the estimate
        idx_tail = np.abs(rho) >= self.rho_thresh
        if np.any(idx_tail):
            orthant[idx_tail] = (1-self.rho_max_w) * orthant[idx_tail]
            orthant_tail = self._orthant_cox_rho(h[idx_tail], k[idx_tail], rho[idx_tail], first_order=True)
            orthant[idx_tail] += self.rho_max_w * orthant_tail
        # Convert to cdf and return
        cdf = orthant_to_cdf(orthant, h, k)
        return cdf
    

    def _orthant_cox_rho(self, h, k, rho, first_order) -> np.ndarray:
        """Wrapper for _orthant_cox_rho_pos to apply both positive and negative rho's"""
        # If h/k/rho not specified, go with attributes
        idx_rho_pos = rho >= 0
        orthant = np.zeros(h.shape)
        if np.any(idx_rho_pos):
            orthant[idx_rho_pos] = self._orthant_cox_rho_pos(h[idx_rho_pos], k[idx_rho_pos], rho[idx_rho_pos], first_order)
        if np.any(~idx_rho_pos):
            orthant[~idx_rho_pos] =  Phi(-h[~idx_rho_pos]) - self._orthant_cox_rho_pos(h[~idx_rho_pos], -k[~idx_rho_pos], -rho[~idx_rho_pos], first_order)
        return orthant
    

    def _orthant_cox_rho_pos(self, h, k, rho, first_order) -> np.ndarray:
        """
        Calculate the cox approximation for positive values of rho
        
        Parameters
        ----------
        ...                     See cdf_cox1
        """
        # Initialize
        orthant = np.zeros(h.shape)
        # (i) At least one positive, rho positive
        idx_pos = (h > 0) | (k > 0)
        if idx_pos.any():
            h1 = np.maximum(h[idx_pos], k[idx_pos])
            k1 = np.minimum(h[idx_pos], k[idx_pos])
            orthant[idx_pos] = self._orthant_cox_base(h1, k1, rho[idx_pos], first_order)
        # (ii) both are negative
        if np.any(~idx_pos):
            h2 = -np.maximum(-h[~idx_pos], -k[~idx_pos])
            k2 = -np.minimum(-h[~idx_pos], -k[~idx_pos])
            orthant[~idx_pos] = 1 - Phi(+h2) - Phi(+k2) + self._orthant_cox_base(-h2, -k2, rho[~idx_pos], first_order)
        return orthant
    

    def _orthant_cox_base(self, h, k, rho, first_order) -> np.ndarray:
        """
        Approximates the orthant probability using inverse mills estimate (see eq. 3 in paper "A Simple Approximation for Bivariate & Triviate Normal Integrals")

        L(h, k, rho) = P(X1 > h)*P(X2 > k | X1 > h)
                    ~ Phi(-h) * Phi((rho*u[h] - k)/sqrt(1-rho^2))
                    = Phi(-h) * Phi(xi(h,k,rho))
                    u[h] = E(X1 | X1 > h)
                    xi(h,k,rho) = (rho*u[h] - k)/sqrt(1-rho^2)
        
        A first order approximation of E[g(Z)] ~ g(E(z)) + 0.5*var(z)*g''(E(z)), leads to a refinement of eq. 3:

        L(h, k, rho) ~ Phi(-h) * [ Phi(xi) - 0.5*rho^2/(1-rho^2)*xi*phi(xi)*sigma^2(h)  ]
                    sigma2(h) = 1 + u[h] - u^2[h]
        
        Parameters
        ----------
        ...                     See cdf_cox1
        first_order:            Whether a first order expansion should be used (see eq. 4), default=False
        """
        # Generate approximations
        u_h = imills(h)
        xi = (rho*u_h - k)/np.sqrt(1-rho**2)
        Phi_h = Phi(-h)
        Phi_xi = Phi(xi)
        if first_order:  # eq. 4
            sigma2_h = 1 + u_h - u_h**2
            orthant = Phi_h * ( Phi_xi - 0.5*rho**2/(1-rho**2)*xi*phi(xi)*sigma2_h )
        else:  # eq. 3
            orthant = Phi_h * Phi_xi
        return orthant

