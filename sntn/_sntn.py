"""
Fully specified SNTN distribution
"""

# External
import numpy as np
from scipy.stats import truncnorm, norm
# Internal
# from sntn.utilities.grad import _log_gauss_approx, _log_diff
# from sntn.utilities.utils import broastcast_max_shape, grad_clip_abs
from sntn._solvers import conf_inf_solver, _process_args_kwargs_flatten


class _SNTN():
    def __init__(self) -> None:
        """
        
        
        Parameters
        ----------
        See _tnorm()

        Attributes
        ----------
        ....

        Methods
        -------
        cdf:            Cumulative distribution function
        pdf:            Density function
        ppf:            Quantile function

        """
        # Input checks and broadcasting
        None


    def cdf(self, x:np.ndarray, **kwargs) -> np.ndarray:
        """Returns the cumulative distribution function"""
        return None


    @staticmethod
    def _dmu_dcdf(mu:np.ndarray, x:np.ndarray, alpha:float, *args, **kwargs) -> np.ndarray:
        """Return the derivative of...."""
        return None

    @staticmethod
    def _find_dist_kwargs(**kwargs) -> tuple:
        """Looks for valid truncated normal distribution keywords Returns sigma, a, b"""
        sigma2, a, b = kwargs['sigma2'], kwargs['a'], kwargs['b']
        kwargs = {k:v for k,v in kwargs.items() if k not in ['sigma2','a','b']}
        return sigma2, a, b, kwargs


    def conf_int(self, x:np.ndarray, alpha:float=0.05, **kwargs) -> np.ndarray:
        """
        Assume X ~ sntn(mu, sigma, a, b), where sigma, a, & b are known. Then for any realized mu_hat (which can be estimated with self.fit()), we want to find the confidence interval for a series of points (x)

        Arguments
        ---------
        x:                      An array-like object of points that corresponds to dimensions of estimated means
        alpha:                  Type-1 error rate
        approx:                 Whether the Gaussian tail approximation should be used
        a_m{in/ax}:             Whether gradient clipping should be used
        fun_x01_type:           Whether a special x to initialization mapping should be applied (default='nudge'). See below. Will be ignored if fun_x{01} is provided
        kwargs:                 For other valid kwargs, see sntn._solvers._conf_int (e.g. a_min/a_max)

        fun_x01_type
        ------------
        nudge:          x0 -> x0, x1 -> 1.01*x1
        bounds:         x0 -> mu_lb, x -> mu_ub
        """
        solver = conf_inf_solver(dist=_tnorm, param_theta='mu',dF_dtheta=self._dmu_dcdf, alpha=alpha)
        # Set up di_dist_args (these go into the tnorm class basically)
        sigma2, a, b, kwargs = self._find_dist_kwargs(**kwargs)
        di_dist_args = {'sigma2':sigma2, 'a':a, 'b':b}
        di_dist_args['approx'] = approx
        # Run CI solver
        res = solver._conf_int(x=x, di_dist_args=di_dist_args, fun_x0=fun_x0, fun_x1=fun_x1, **kwargs)
        # Return matrix of values
        return res
