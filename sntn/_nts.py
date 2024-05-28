"""
Fully specified SNTN distribution
"""

# External
import numpy as np
from time import time
from warnings import warn
from scipy.optimize import root
from scipy.stats import truncnorm, norm
# Internal
from sntn._bvn import _bvn
from sntn._solvers import conf_inf_solver
from sntn.utilities.grad import _log_gauss_approx
from sntn._fast_integrals import _rootfinder_newton, bvn_cdf_diff
from sntn.utilities.utils import broastcast_max_shape, try2array, \
        broadcast_to_k, reverse_broadcast_from_k, \
        pass_kwargs_to_classes, get_valid_kwargs_cls, \
        get_valid_kwargs_func, get_valid_kwargs_method, \
        vprint

@staticmethod
def _mus_are_equal(mu1, mu2) -> tuple:
    """When we desire to fix mu1/mu2 to the same value, will return the value of both when one of them is None"""
    # Determine which are None (if any)
    mu1_is_None, mu2_is_None = False, False
    if isinstance(mu1, np.ndarray):
        mu1_is_None = None in mu1
    else:
        mu1_is_None = mu1 is None
    if isinstance(mu2, np.ndarray):
        mu2_is_None = None in mu2
    else:
        mu2_is_None = mu2 is None
    # Return the appropriate value
    if mu1_is_None:
        assert not mu2_is_None, 'if mu1 is None, mu2 cannot be'
        return mu2, mu2
    elif mu2_is_None:
        assert not mu1_is_None, 'if mu2 is None, mu1 cannot be'
        return mu1, mu1
    else:
        assert np.all(mu1 == mu2), 'if mu is fixed, mu1 != mu2'
        return mu1, mu2


class _nts():
    def __init__(self, mu1:float | np.ndarray | None, tau21:float | np.ndarray, mu2:float | np.ndarray | None, tau22:float | np.ndarray, a:float | np.ndarray, b:float | np.ndarray, c1:float | np.ndarray=1, c2:float | np.ndarray=1, fix_mu:bool=False, **kwargs) -> None:
        """
        The "normal and truncated sum": workhorse class for the sum of a normal and truncated normal. Carries out standard inferece using scipy.dist syntax with added conf_int method

        W = c1*Z1 + c2*Z2,  Z1 ~ N(mu1, tau21), Z2 ~ TN(mu2, tau22, a, b)
        W ~ NTS(theta(mu(c)), Sigma(tau(c)), a, b)
        mu(c) =             [c1*mu1, c2*mu2]
        tau(c) =            [c1**2 * tau21, c2**2 * tau22]
        theta(mu(c)) =      [sum(mu), mu[1]]
        sigma2 =            [sum(tau), tau[1]]
        sigma =             sqrt(sigma2)
        rho =               sigma[1] / sigma[0]
        Sigma(tau(c)) =     [ [sigma2[0], sigma2[1]], [sigma2[1], sigma2[1]]]
        
        
        Parameters
        ----------
        mu1:             Mean of the unconditional gaussian (can be None is fix_mu=True)
        tau21:           The variance of the unconditional gaussian (must be strictly positive)
        mu2:             The mean of the truncated Gaussian (can be None is fix_mu=True)
        tau22:           The variance of the unconditional gaussian (must be strictly positive)
        a:               The lowerbound of the truncated Gaussian (must be smaller than b)
        b:               The upperbound of the truncated Gaussian (must be larger than a)
        c1:              A weighting constant for Z1 (must be >0, default=1)
        c2:              A weighting constant for Z2 (must be >0, default=1)
        fix_mu:          Whether mu1=mu2=mu, will need mu1 OR mu2 to be None, but not both (default=False)
        kwargs:          Other keywords to be passed onto the bvn() construction

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
        if isinstance(fix_mu, np.ndarray):
            fix_mu = bool(fix_mu.flat[0])  # Assume it was broadcasted
        assert isinstance(fix_mu, bool), 'fix_mu needs to be a boolean'
        if fix_mu:
            mu1, mu2 = _mus_are_equal(mu1, mu2)
        self.fix_mu = fix_mu
        mu1, mu2, tau21, tau22, a, b, c1, c2 = broastcast_max_shape(mu1, mu2, tau21, tau22, a, b, c1, c2)
        assert np.all(tau21 > 0), 'tau21 needs to be strictly greater than zero'
        assert np.all(tau22 > 0), 'tau22 needs to be strictly greater than zero'
        assert np.all(b > a), 'b needs to be greated than a'
        # Capture the original shape for later transformations
        self.param_shape = mu1.shape
        # Flatten parameters
        mu1, mu2, tau21, tau22, a, b, c1, c2 = [x.flatten() for x in [mu1, mu2, tau21, tau22, a, b, c1, c2]]
        # Store the original attributes
        self.mu1, self.c1, self.tau21 = mu1, c1, tau21
        self.mu2, self.c2, self.tau22 = mu2, c2, tau22
        self.a, self.b = a, b
        self.k = len(mu1)

        # Create new attributes
        self.theta1 = c1*mu1 + c2*mu2
        self.theta2 = mu2
        sigma21 = c1**2 * tau21 + c2**2 * tau22
        sigma22 = tau22
        self.sigma1 = np.sqrt(sigma21)
        self.sigma2 = np.sqrt(sigma22)
        self.rho = c2 * self.sigma2 / self.sigma1
        # Calculate the truncated normal terms
        self.alpha = (a - mu2) / np.sqrt(tau22)
        self.beta = (b - mu2) / np.sqrt(tau22)
        # Calculate Z, use 
        self.Z = norm.cdf(self.beta) - norm.cdf(self.alpha)
        # If we get tail values, true the approx
        idx_tail = (self.Z == 0) | (self.Z == 1)
        if idx_tail.any():
            self.Z[idx_tail] = np.exp(_log_gauss_approx(self.beta[idx_tail], self.alpha[idx_tail]))
        # Initialize normal and trunctated normal
        self.dist_Z1 = norm(loc=mu1, scale=np.sqrt(tau21))
        self.dist_Z2 = truncnorm(loc=mu2, scale=np.sqrt(tau22), a=self.alpha, b=self.beta)
        # Create the bivariate normal distribution
        self.bvn = pass_kwargs_to_classes(_bvn, 0, 1, 0, 1, self.rho, **kwargs)


    def mean(self) -> np.ndarray:
        """Calculate the mean of the NTS distribution"""
        mu = self.c1*self.dist_Z1.mean() + self.c2*self.dist_Z2.mean()
        mu = mu.reshape(self.param_shape)
        return mu


    def cdf(self, w: np.ndarray, 
            method: str = 'bvn', 
            clip: float = 30, 
            **kwargs
        ) -> np.ndarray:
        """
        Returns the cumulative distribution function
        w: np.ndarray
        
        Inputs
        ======
        method: str
            Which CDF calculation method to use; see Methods (default = 'bvn')
        clip: float
            For the 'fast' method, bounds the beta/alpha to between ±clip (default = 30)
        **kwargs
            Any other arguments to pass to the _bvn class construction
        
        Methods
        =======
        bvn: Uses one of the BVN CDF methods (see sntn._bvn._bvn)
        fast: Uses the sntn._fast_integrals.bvn_cdf_diff method
        """
        # Consider updates bvn with kwargs match
        cdf_methods = ['bvn', 'fast']
        assert method in cdf_methods, f'method must be one of {cdf_methods}'
        # Broadcast x to the same dimension of the parameters
        w = broadcast_to_k(np.atleast_1d(w), self.param_shape)
        m1 = (w - self.theta1) / self.sigma1
        if method == 'bvn':
            kwargs_bvn = get_valid_kwargs_cls(_bvn, **kwargs)
            if len(kwargs_bvn) > 0:
                self.bvn = pass_kwargs_to_classes(_bvn, 0, 1, 0, 1, self.rho, **kwargs_bvn)
            # Calculate the CDF (note that 1-(orth1-orth2)/Z = (CDF2 - CDF1)/Z)
            cdf1 = self.bvn.cdf(x1=m1, x2=self.alpha)
            cdf2 = self.bvn.cdf(x1=m1, x2=self.beta)
            pval = (cdf2 - cdf1) / self.Z
            # Do some cleanup for the tails
            # If cdf2 and cdf1 are ~100%, then tail is so extreme solution is one
            pval[(cdf2 - cdf1 == 0) & (cdf2.round() == 1)] = 1
            # If Z is zero, then it's going to be zero
            pval[(cdf2 == 0) & (cdf1 == 0) & (self.Z == 0)] = 0
        if method == 'fast':
            beta = np.clip(self.beta, a_min=None, a_max=+clip)
            alpha = np.clip(self.alpha, a_min=-clip, a_max=None)
            pval = bvn_cdf_diff(x1=m1, x2a=beta, x2b=alpha, rho=self.rho) / self.Z
        # Return to proper shape
        pval = reverse_broadcast_from_k(pval, self.param_shape)
        # Bound b/w [0,1]
        pval = np.clip(pval, 0, 1)
        return pval
            

    def pdf(self, x:np.ndarray) -> np.ndarray:
        """Calculates the marginal density of the NTS distribution at some point x"""
        x = try2array(x)
        x = broadcast_to_k(x, self.param_shape)
        # Calculate pdf
        term1 = self.sigma1 * self.Z
        m1 = (x - self.theta1) / self.sigma1
        term2 = (self.beta-self.rho*m1) / np.sqrt(1-self.rho**2)
        term3 = (self.alpha-self.rho*m1) / np.sqrt(1-self.rho**2)
        f = norm.pdf(m1)*(norm.cdf(term2) - norm.cdf(term3)) / term1
        # Return
        f = reverse_broadcast_from_k(f, self.param_shape)
        return f


    def rvs(self, ndraw:int, seed=None) -> np.ndarray:
        """Generate n samples from the distribution"""
        z1 = self.dist_Z1.rvs([ndraw,self.k], random_state=seed)
        z2 = self.dist_Z2.rvs([ndraw, self.k], random_state=seed)
        w = self.c1*z1 + self.c2*z2
        w = reverse_broadcast_from_k(w, self.param_shape)
        return w


    def _err_cdf_p(self, w:np.ndarray, p:np.ndarray) -> np.ndarray:
        """
        Utility function that can be passed into a root solver so that a (n,k) array of points can be evaluated to an (n,k) array of percentiles

        When w/p are flat arrays, we assume assume they need to be reshaped to be passed into the cdf function. When they have two or more dimensions, we assume that the output needs to be flattenerd
        """
        assert w.shape == p.shape, 'Unsure why w.shape != p.shape?'
        if len(w.shape) == 1:
            # w/p are the incorrect shapes for evaluation, they need to be reshaped to be (n,k)
            n = w.shape[0] / self.k
            assert n == int(n), 'expected n to be a whole number'
            n = int(n)
            #w_r = np.squeeze(w.reshape([n, self.k]))
            w_r = np.squeeze(w.reshape([n, *self.param_shape]))
            pval = np.squeeze(self.cdf(w_r))
            p = np.squeeze(p.reshape(pval.shape))
            err = pval - p
        else:
            # w/p are the correct shapes for evaluation, but they need to be flattened for the root solver
            err = self.cdf(w) - p
        err = np.atleast_1d(err).flatten()
        return err


    def ppf(self, 
            p: np.ndarray, 
            method: str = 'fast', 
            tol_cdf: float = 1e-3, 
            verbose: bool = False, 
            verbose_iter: int = 50,
            root_iter: int | None = None,
            use_approx_init: bool = True,
            clip: float = 30,
            calc_rvs_init: bool = True,
            seed: int = 1234,
            n_samp: int = 1000,
            **kwargs
        ) -> np.ndarray:
        """
        Returns the quantile of the NTS distribution(s)
        
        Arguments
        ---------
        p: np.ndarray
            Percent quantile we are looking to find, w(p), so that F(w) = p
        method: str
            Which quantile-finding method should be used? See "Methods" below, defaults to 'fast'
        tol_cdf: float=1e-3
            After w(p) is found, check maximum error b/w |F(w) - p| < tol_cdf
        verbose: bool
            During the loop, should the iteration be printed? (default==False)
        verbose_iter: int
            During the loop, at which iteration should the print occur (default==50)
        root_iter: int | None
            For method=='root', how much roots should we solve at the same time? This is useful for an array of quantiles. Note that the final count will be between: (k*(iter//k) ,self.k). Defaults to self.k if is None
        calc_rvs_init: bool = True
            Should RVS be used to calculate an initial guestimate?
        seed: int = 1234
            If RVS is used, what seed should be used?
        n_samp: int = 1000
            How many samples are needed? Will be max(n_samp, len(p))
        use_approx_init: bool
            For the 'fast' method, should the 'approx' weights be initialized, or use the default from _fast_integrals? (default==True)
        clip: float
            For the 'fast' method, how to limit +- infinity to some large nubmer (default = 30)
        **kwargs           
            Will be passed onto _rootfinder_newton (consider 'use_gradclip', 'clip_low', or 'clip_high' for convergence failures)
        
        Methods
        -------
        fast:               Solves all roots simultaneously using Newton's method (unstable for small Z)
        root:               Solve all roots simultaneously (fast but unstable)
        loop:               Loop over all i,j configurations (slower but more stable)
        approx:             Use the quantiles from each dist (i.e. c1*norm.ppf(alpha) + c2*tnorm(alpha)
        rvs:                Use simulated data
        """
        valid_ppf_methods = ['fast', 'root', 'approx', 'loop', 'rvs']
        assert method in valid_ppf_methods, f'method must be one of {valid_ppf_methods}'
        # Make sure aligns with the parameters
        orig_p = np.atleast_1d(p)
        assert (len(orig_p.shape) == 1) | (orig_p.shape == self.param_shape), f'currently the ppf method only supports broadcasting an array of percentiles, or matching the original parameter shape: {self.param_shape}. Your shape = {orig_p.shape}'
        num_p = len(orig_p)
        p = broadcast_to_k(orig_p, self.param_shape)
        if use_approx_init and method == 'fast':  # Override the RVS
            calc_rvs_init = False
        if calc_rvs_init:
            n_samp = max(n_samp, num_p)
            W_sample = self.rvs(n_samp, seed)
            w0 = np.quantile(W_sample, q=orig_p, axis=0)
            w0 = broadcast_to_k(w0, self.param_shape)
        else:
            w0 = self.c1*self.dist_Z1.ppf(p) + self.c2*self.dist_Z2.ppf(p)
        assert p.shape == w0.shape, 'Expected ppf of dist_Z{12} to align with p shape'
        if method == 'fast':
            # Solve in the m(w) space
            kwargs_newton = get_valid_kwargs_func(_rootfinder_newton, **kwargs)
            # Broadcast the parameters
            target_p = np.squeeze(self.Z * p)  # If it can be flat, let it be
            target_p, beta, alpha, rho, sigma1, theta1, Zphi = np.broadcast_arrays(target_p, self.beta, self.alpha, self.rho, self.sigma1, self.theta1, self.Z)
            # Clip alpha and beta
            beta = np.clip(beta, a_min=None, a_max=+clip)
            alpha = np.clip(alpha, a_min=-clip, a_max=None)
            # Run the root-finder and (possibly) set the initial values
            if use_approx_init:
                w_init = np.broadcast_to(np.squeeze(w0), shape=theta1.shape)
                m_init = (w_init - theta1) / sigma1
                kwargs_newton['x0_vec'] = m_init
            m_roots = _rootfinder_newton(ub=beta, lb=alpha, rho=rho, target_p=target_p, **kwargs_newton)
            # Solve for w
            w = m_roots * sigma1 + theta1
            cdf_roots = bvn_cdf_diff(x1=m_roots, x2a=beta, x2b=alpha, rho=rho) / Zphi
            cdf_roots = np.clip(cdf_roots, 0, 1)
            err_cdf = np.abs(cdf_roots - p)
            # Identify any failures
            w = np.broadcast_to(w, err_cdf.shape).copy()
            if err_cdf.max() > tol_cdf:
                idx_err = err_cdf > tol_cdf
                warn(f'Heads ups, a total of {idx_err.sum()} roots of {np.prod(idx_err.shape)} could not be solved, using random draws to approximate, adjust the n_samp argument as needed')
                # There parameters have at least one erro
                idx_err_params = np.any(idx_err, axis=0)
                full_mask = np.zeros_like(p, dtype=bool)  # Initialize a mask of the same shape as w
                full_mask[:, idx_err_params] = idx_err[:, idx_err_params]  # Apply error flags based on your conditions
                tmp_sntn = _nts(mu1=self.mu1[idx_err_params], mu2=self.mu2[idx_err_params], 
                                    tau21=self.tau21[idx_err_params], tau22=self.tau22[idx_err_params], 
                                    a=self.a[idx_err_params], b=self.b[idx_err_params], 
                                    c1=self.c1[idx_err_params], x2=self.c2[idx_err_params], fix_mu=self.fix_mu)
                # Calculate the quantiles (some may be superfluous if there's no error)
                w_err = np.quantile(tmp_sntn.rvs(n_samp, seed), orig_p, axis=0)
                # Overwrite the relevant indices
                w[full_mask] = w_err[full_mask[:, idx_err_params]]
            # Reshape for reverse_broadcast_from_k
            w = w.reshape(w0.shape)
        if method == 'rvs':
            w = np.quantile(self.rvs(n_samp, seed), orig_p, axis=0)
            w = w.reshape(w0.shape)
        if method == 'approx':
            # Use the simple quantiles
            w = w0.copy()
        if method == 'root':
            # err_cdf_p needs to be flattened
            w0_flat, p_flat = w0.flatten(), p.flatten()
            # How many calculations can we get per iteration (max of root_iter)
            if root_iter is None:
                root_iter = self.k
            iter_act = self.k * (root_iter // self.k)
            n_loop = int(np.ceil(self.k * num_p / iter_act))
            vprint(f'Calculating {iter_act} roots per iteration over {n_loop} loops', verbose)
            # Loop over all solutions
            kwargs_root = get_valid_kwargs_func(root, **kwargs)
            solution = np.zeros(w0_flat.shape)
            for loop in range(n_loop):
                # Break up into batches of at most 
                idx_low, idx_high = iter_act*loop, iter_act*(loop+1)
                w0_loop, p_loop = w0_flat[idx_low:idx_high], p_flat[idx_low: idx_high]
                solroot = root(self._err_cdf_p, w0_loop, args=(p_loop, ), **kwargs_root)
                if (loop+1) == n_loop:
                    solution[idx_low:] = solroot.x  # Last iteration may have a different shape
                else:
                    solution[idx_low:idx_high] = solroot.x
                merr = np.abs(solroot.fun).max()
                if merr > tol_cdf:
                    warn(f'Error! Root finding had a max error {merr} which exceeded tolerance {tol_cdf} at loop iteration {loop} of {n_loop}\n\nTRY LOWERING root_iter=={self.k}')
            w = solution.reshape(w0.shape) # Reshape
        if method == 'loop':
            # Prepare for loop
            w0, p = np.atleast_2d(w0), np.atleast_2d(p)
            n = len(w0)
            ntot = int(np.prod(w0.shape))
            w = np.zeros(w0.shape)
            stime = time()
            # Outer loop if the j'th parameter (out of k)
            for j in range(self.k):
                dist_j = _nts(self.mu1[j], self.tau21[j], self.mu2[j], self.tau22[j], self.a[j], self.b[j], self.c1[j], self.c2[j], cdf_approach=self.bvn.cdf_approach)
                fun_j = lambda xx, pp: dist_j.cdf(xx) - pp            
                # Inner loop are the n quantile point
                for i in range(n):
                    w0_ij = w0[i,j]
                    p_ij = p[i,j]
                    solution_ij = root(fun=fun_j, x0=w0_ij, args=(p_ij))
                    merr_ij = np.max(np.abs(solution_ij.fun))
                    if merr_ij > tol_cdf:
                        warn(f'Error {merr_ij:.5f} > {tol_cdf:.5f} at iteration i={i}, j={j}, trying alternative solution')
                        if j >= 1:
                            # Try the previous solutions (works best when p_seq has a small space)
                            solution_ij = root(fun=fun_j, x0=w[i,j-1], args=(p_ij))
                            if np.max(np.abs(solution_ij.fun)) < tol_cdf:
                                continue
                    w[i,j] = solution_ij.x[0]
                    if verbose:
                        ncomp = i*self.k + (j+1)
                        if ncomp % verbose_iter == 0:
                            dtime = time() - stime
                            nleft = ntot - ncomp
                            rate = ncomp / dtime
                            seta = nleft / rate
                            print(f'Iteration {ncomp} of {ntot} (ETA={seta/60:.1f} minutes)')
        # Put to original param shape
        w = reverse_broadcast_from_k(w, self.param_shape)
        return w


    def _find_dist_kwargs_CI(self, **kwargs) -> tuple:
        """
        Looks for valid truncated normal distribution keywords that are needed for generating a CI

        Returns
        -------
        {mu1,mu2}, tau21, tau22, c1, c2, fix_mu, kwargs
        """
        valid_kwargs = ['mu1', 'mu2', 'tau21', 'tau22', 'c1', 'c2', 'fix_mu']
        # tau21/tau22 are always required
        tau21, tau22 = kwargs['tau21'], kwargs['tau22']
        # If c1/c2/fix_mu are not specified, assume they are the default values
        if 'c1' not in kwargs:
            c1 = 1
        if 'c2' not in kwargs:
            c2 = 1
        if 'fix_mu' not in kwargs:
            fix_mu = self.fix_mu
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
        return mu, tau21, tau22, c1, c2, fix_mu, kwargs


    def conf_int(self, x:np.ndarray, alpha:float=0.05, n_chunks:int=1, param_fixed:str='mu', **kwargs) -> np.ndarray:
        """
        Assume W ~ NTS()...

        Confidence intervals for the NTS distribution are slightly different than other distribution because theta is a function of mu1/mu2. Instead we can do CIs for mu1 or mu2, or mu=mu1=mu2 (see fix_mu in the constructor)

        Arguments
        ---------
        x:                      An array-like object of points that corresponds to dimensions of estimated means
        alpha:                  Type-1 error rate
        param_fixed:            Which parameter are we doing inference on ('mu'==fix mu1==mu2, 'mu1', 'mu2')?
        n_chunks:               How many roots to solve at a time? (default=1) 
        kwargs:                 For other valid kwargs, see sntn._solvers._conf_int (e.g. a_min/a_max)
        """
        # Make sure x is the right dimension
        x = try2array(x)
        x = broadcast_to_k(x, self.param_shape)
        if x.shape == (self.k,):
            # Assume each point corresponds to a single parameter
            x = np.expand_dims(x, 0)

        # Storage for the named parameters what will go into class initialization (excluded param_fixed)
        di_dist_args = {}
        # param_fixed must be either mu1, mu2, or mu (mu1==mu2)
        valid_fixed = ['mu1', 'mu2', 'mu']
        assert param_fixed in valid_fixed, f'param_fixed must be one of: {param_fixed}'
        if param_fixed == 'mu':
            param_fixed = 'mu1'  # Use mu1 for convenience
            assert self.fix_mu==True, 'if param_fixed="mu" then fix_mu==True'
            di_dist_args['mu2'] = None  # One parameter must be None with fix_mu=True, and since mu1 will be assigned every time, this is necessary
        elif param_fixed == 'mu1':
            param_fixed = 'mu1'
            di_dist_args['mu2'] = self.mu2
        else:
            param_fixed = 'mu2'
            di_dist_args['mu1'] = self.mu1
            # x_mu1, x_mu2 = self.mu1, x/2

        # Set up solver along with kwargs
        solver = conf_inf_solver(dist=_nts, param_theta=param_fixed, alpha=alpha, **get_valid_kwargs_cls(conf_inf_solver, **kwargs))
        # Assign the remainder of the parameter
        di_dist_args = {**di_dist_args, **{'tau21':self.tau21, 'tau22':self.tau22, 'a':self.a, 'b':self.b, 'c1':self.c1, 'c2':self.c2, 'fix_mu':self.fix_mu}}
        # Run CI solver
        verbose, verbose_iter = False, 50
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            assert isinstance(verbose, bool)
        if 'verbose_iter' in kwargs:
            verbose_iter = kwargs['verbose_iter']
            assert isinstance(verbose_iter, int) and (verbose_iter > 0)
        if 'cdf_approach' in kwargs:
            di_dist_args['cdf_approach'] = kwargs['cdf_approach']
        # Set up solver kwargs
        kwargs_for_conf_int = get_valid_kwargs_method(solver, '_conf_int', **kwargs)
        
        # Iterate over each parameter
        res = np.zeros(x.shape + (2,))
        n_x = x.shape[0]
        n_iter = (n_x // n_chunks) + (n_x % n_chunks)
        n_total = self.k*n_iter  # Total number of .cont_int calls
        stime = time()
        for i in range(self.k):
            # Take the i'th parameter values
            di_dist_args_i = {k:v[i] if isinstance(v, np.ndarray) else v for k,v in di_dist_args.items()}
            x_i = np.atleast_1d(np.take(x, i, -1))
            # Loop over the n_iter rows of chunk
            for j in range(n_iter):
                start, stop = int(j*n_chunks), int((j+1)*n_chunks)
                x_ij = x_i[start:stop]
                ci_ij = solver._conf_int(x=x_ij, di_dist_args=di_dist_args_i, x0=x_ij, **kwargs_for_conf_int)
                res[start:stop,...,i,:] = ci_ij
                if verbose and ((j+1) % verbose_iter == 0):
                    dtime = time() - stime
                    n_comp = i*n_iter + (j+1)
                    n_left = n_total - n_comp
                    rate = n_comp / dtime
                    seta = n_left / rate
                    print(f'Parameter {i+1} of {self.k}, Iteration {j+1} of {n_iter}\n(ETA={seta:0.0f} seconds)')

        # Return to original shape
        res = reverse_broadcast_from_k(res, self.param_shape,(2,))
        return res
