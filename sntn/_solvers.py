"""
Words
"""


def get_CI(self, x:np.ndarray, approach:str, alpha:float=0.05, approx:bool=True, a_min:None or float=0.005, a_max:None or float=None, mu_lb:float or int=-100000, mu_ub:float or int=100000, **kwargs) -> np.ndarray:
    """
    Assume X ~ TN(mu, sigma, a, b), where sigma, a, & b are known. Then for any realized mu_hat (which can be estimated with self.fit()), we want to find 
    Calculate the confidence interval for a series of points (x)

    Parameters
    ----------
    x:                  An array-like object of points that corresponds to dimensions of estimated means
    approach:           A total of four approaches have been implemented to calculate the CIs (default=root_scalar, see scipy.optimize.{root, minimize_scalar, minimize, root_scalar})
    alpha:              Type-I error for the CIs (default=0.05)
    mu_lb:              Found bounded optimization methods, what is the lower-bound of means that will be considered for the lower-bound CI?
    mu_ub:              Found bounded optimization methods, what is the upper-bound of means that will be considered for the lower-bound CI?
    kwargs:             Named arguments which will be passed into the scipy.optims (default={'method':'secant'} if approach is default)

    Notes on optimization approaches
    -----
    Optimal approaches
    approach            method          notes
    root_scalar         secant          Optimal speed/accuracy
    minimize_scalar     Golden          Slower, but best accuracy
    
    approach            method          Recommended
    minimize            BFGS            False
                        COBYLA          False
                        L-BFGS-B        False
                        Nelder-Mead     False
                        Powell          True
                        SLSQP           False
                        TNC             False
    minimize_scalar     Bounded         False
                        Brent           True
                        Golden          True
    root                hybr            False
                        lm              False
    root_scalar         bisect          True
                        brenth          True
                        brentq          True
                        newton          False
                        ridder          True
                        secant          True
                        toms748         True

    Returns
    -------
    An ({x.shape,mu.shape},2) array for the lower/upper bound. Shape be different than x if x gets broadcasted by the existing parameters
    """
    # Input checks
    valid_approaches = ['root', 'minimize_scalar', 'minimize', 'root_scalar']
    assert approach in valid_approaches, f'approach needs to be one of {valid_approaches}'
    # Try to broadcast x to match underlying parameters
    # Guess some lower/upper bounds
    c_alpha = norm.ppf(alpha/2)
    x0_lb, x0_ub = self.mu + c_alpha, self.mu - c_alpha
    # Squeeze x if we can in cause parameters are (n,)
    x = np.squeeze(x)
    x, mu, sigma, a, b, x0_lb, x0_ub = broastcast_max_shape(x, self.mu, self.sigma, self.a, self.b, x0_lb, x0_ub)
    
    # Set a default "method" for each if not supplied in the kwargs
    di_default_methods = {'root':'hybr',
                            'minimize_scalar':'Golden',
                            'minimize':'Powell',
                            'root_scalar':'secant'}
    if 'method' not in kwargs:
        kwargs['method'] = di_default_methods[approach]

    # Set up the argument for the vectorized vs scalar methods
    if approach in ['minimize_scalar','root_scalar']:
        # x, sigma, a, b, alpha, approx, a_min, a_max
        solver_args = [x.flat[0], sigma.flat[0], a.flat[0], b.flat[0], 1-alpha/2, approx, a_min, a_max]
        # Will be assigned iteratively
        ci_lb = np.full_like(mu, fill_value=np.nan)
        ci_ub = ci_lb.copy()
    else:
        should_flatten = True
        solver_args = [x.flatten(), sigma.flatten(), a.flatten(), b.flatten(), alpha, approx, a_min, a_max, should_flatten]

    if approach == 'root_scalar':
        # ---- Approach #1: Point-wise root finding ---- #
        # There are four different approaches to configure root_scalar (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
        di_root_scalar_extra = {}
        if kwargs['method'] in ['bisect', 'brentq', 'brenth', 'ridder','toms748']:
            di_root_scalar_extra['bracket'] = (mu_lb, mu_ub)
        elif kwargs['method'] == 'newton':
            di_root_scalar_extra['x0'] = None
            di_root_scalar_extra['fprime'] = self._dmu_dcdf
        elif kwargs['method'] == 'secant':
            di_root_scalar_extra['x0'] = None
            di_root_scalar_extra['x1'] = None
        elif kwargs['method'] == 'halley':
            di_root_scalar_extra['x0'] = None
            di_root_scalar_extra['fprime'] = self._dmu_dcdf
            di_root_scalar_extra['fprime2'] = self._dmu_dcdf

        # Prepare the parts of the optimization that won't change (note that kwargs will overwrite di_base)
        di_base = {'f':self._err_cdf0, 'xtol':1e-4, 'maxiter':250}  # Hard-coding unless specified by user based on stability experiments
        di_base = {**di_base, **kwargs}
        # Loop over all element points
        for kk in np.ndindex(x.shape): 
            # Update the solver_args
            mu_kk, x_kk, sigma_kk, a_kk, b_kk = mu[kk], x[kk], sigma[kk], a[kk], b[kk]
            solver_args[:4] = x_kk, sigma_kk, a_kk, b_kk
            # Extra kk'th element
            if 'x0' in di_root_scalar_extra:
                di_root_scalar_extra['x0'] = mu_kk
            if 'x1' in di_root_scalar_extra:
                di_root_scalar_extra['x1'] = x_kk
            # Hard-coding iterations
            di_lb = {**di_base, **{'args':solver_args}, **di_root_scalar_extra}
            di_ub = deepcopy(di_lb)
            di_ub['args'][4] = alpha/2
            di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
            lb_kk = float(root_scalar(**di_lb).root)
            ub_kk = float(root_scalar(**di_ub).root)
            ci_lb[kk] = lb_kk
            ci_ub[kk] = ub_kk            

    elif approach == 'minimize_scalar':
        # ---- Approach #2: Point-wise gradient ---- #
        di_base = {'fun':self._err_cdf2, 'bounds':(mu_lb, mu_ub)}
        di_base = {**di_base, **kwargs}
        # Loop over all element points
        for kk in np.ndindex(x.shape):
            mu_kk, x_kk, sigma_kk, a_kk, b_kk = mu[kk], x[kk], sigma[kk], a[kk], b[kk]
            solver_args[:4] = x_kk, sigma_kk, a_kk, b_kk
            di_lb = {**di_base, **{'args':solver_args}}
            di_ub = deepcopy(di_lb)
            di_ub['args'][4] = alpha/2
            di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
            lb_kk = minimize_scalar(**di_lb).x
            ub_kk = minimize_scalar(**di_ub).x
            ci_lb[kk] = lb_kk
            ci_ub[kk] = ub_kk
    
    elif approach == 'minimize':
        # ---- Approach #3: Vector gradient ---- #
        di_base = {**{'fun':self._err_cdf2, 'args':solver_args, 'x0':mu.flatten()},**kwargs}
        if kwargs['method'] in ['CG','BFGS','L-BFGS-B','TNC','SLSQP']:
            # Gradient methods
            di_base['jac'] = self._derr_cdf2
        elif kwargs['method'] in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
            raise Warning(f"The method you have specified ({kwargs['method']}) is not supported")
            return None
        else:
            # Gradient free methods
            assert kwargs['method'] in ['Nelder-Mead', 'Powell', 'COBYLA']
        # Run
        di_lb, di_ub = deepcopy(di_base), deepcopy(di_base)
        di_lb['args'][4] = 1-alpha/2
        di_ub['args'][4] = alpha/2
        di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
        ci_lb = minimize(**di_lb).x
        ci_ub = minimize(**di_ub).x
        # Return to original size
        ci_lb = ci_lb.reshape(x.shape)
        ci_ub = ci_ub.reshape(x.shape)

    else:
        # ---- Approach #4: Vectorized root finding ---- #
        di_base = {**{'fun':self._err_cdf, 'jac':self._dmu_dcdf, 'args':solver_args},**kwargs}
        if kwargs['method'] in ['broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']:
            # raise Warning(f"The method you have specified ({kwargs['method']}) is not supported")
            # return None
            del di_base['jac']
        else:
            assert kwargs['method'] in ['hybr', 'lm']
        di_lb, di_ub = deepcopy(di_base), deepcopy(di_base)
        di_lb['args'][4] = 1-alpha/2
        di_lb['x0'] = x0_lb.flatten()
        di_ub['args'][4] = alpha/2
        di_ub['x0'] = x0_ub.flatten()
        di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
        ci_lb = root(**di_lb).x
        ci_ub = root(**di_ub).x
    # Return values
    mat = np.stack((ci_lb,ci_ub),ci_lb.ndim)
    return mat 