"""
Contains the base class conf_inf_solver which can be used to do root-finding for generating confidence intervals for an arbitrary distribution function.

Notes on optimization approaches
-----
Other methods can be used, although any approach that is ~Recommended cannot guarantee convergence
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

"""

# External 
import numpy as np
from time import time
from warnings import warn
from inspect import getfullargspec
from typing import Callable, Optional
from scipy.optimize import root, minimize_scalar, minimize, root_scalar
# Internal modules
from sntn.utilities.utils import broastcast_max_shape, str2list, no_diff

# Hard-coded scipy approaches and methods
valid_approaches = ['root', 'minimize_scalar', 'minimize', 'root_scalar']
# First item is "recommended"
di_default_methods = {'root':['hybr', 'lm'],
                      'minimize_scalar':['Golden', 'Bounded', 'Brent'],
                      'minimize':['Powell','COBYLA','L-BFGS-B'],
                      'root_scalar':['secant','bisect', 'brentq', 'brenth', 'ridder','toms748','newton']}
no_diff(valid_approaches, di_default_methods.keys())


@staticmethod
def _return_x01_funs(fun_x01_type:str='nudge', **kwargs) -> tuple:
    """Return the two fun_x{01} to be based into CI solver to find initialization points"""
    valid_types = ['nudge']
    assert fun_x01_type in valid_types, f'If fun_x01_type is specified it must be one of {valid_types}'
    if fun_x01_type == 'nudge':
        fun_x0 = lambda x: np.atleast_1d(x) * 0.99
        fun_x1 = lambda x: np.atleast_1d(x) * 1.01
    return fun_x0, fun_x1


@staticmethod
def _check_flatten(**kwargs) -> tuple:
    """
    Used by the _err_cdf and _err_cdf2 functions to see if there is a 'flatten' argument that gets passed into dist_kwargs. Will return the boolean value of flatten, and then remove the key from the dict so as to not cause an error
    """
    # Check if we will flatten
    flatten = False
    if 'flatten' in kwargs:
        flatten = kwargs['flatten']
        assert isinstance(flatten, bool), 'If flatten is added to **dist_kwargs, then it needs to a bool'
        del kwargs['flatten']  # deepcopy not needed since passed in as **kwargs rather than kwargs:dict
    return flatten, kwargs


@staticmethod
def _process_args_kwargs_flatten(args:tuple, kwargs:tuple) -> tuple:
    """For methods that take in either an args/kwargs, processes the args as though they were kwargs, assuming that the first element in the args if the named arguments. If args is empty, will only use kwargs. Also runs a flatten check."""
    if len(args) > 0:
        di_names = str2list(args[0])
        if len(di_names) > 1:
            # Will be a list of lists
            kwargs = dict(zip(di_names, args[1:][0]))
        else:
            # Should zip fine
            kwargs = dict(zip(di_names, args[1:]))
    # Return the flatten variable along with the kwargs
    return _check_flatten(**kwargs)


class conf_inf_solver():
    def __init__(self, dist:callable, param_theta:str, dF_dtheta : Optional[Callable] = None, alpha:float=0.05, verbose:bool=False, verbose_iter:int=50) -> None:
        """
        The conf_inf_solver class can generate confidence intervals for a single parameter of a distribution. Specifically for a target parameter "theta":

        CI_ub: inf_theta: F(theta;phi1,...,phik).cdf(x)-alpha/2=0
        CI_lb: sup_theta: F(theta;phi1,...,phik).cdf(x)-1+alpha/2=0
        whhere phi1,...,phik are other parameters of the distribution

        Note that the ub/lb distribution may be reversed depending on if dF/dtheta is strictly positive or negative. For example, if the Exponential distribution is parameterized by the "scale" (1/"rate") then the CI_ub will be smaller than the CI_lb.
        

        Parameters
        ----------
        dist:               A distribution function that has a "cdf" method and can be constructed with named parameters (e.g. scipy.stats.norm(loc=1,scale=[2,3]))
        param_theta:        Named argument (str) of the paramater of interested with the function dist({param_theta}=...)
        dF_dtheta:          Calculates the derivative of the CDF w.r.t. theta, must have named arguments (x=,alpha=,{param_theta}=,{di_dist_args}=) (OPTIONAL, default=None)
        alpha:              Type-I error rate (default=0.05)


        Methods
        -------
        _err_cdf:            Used to calculate the difference CDF and type-1 error rate
        _err_cdf0:           Returns float is evaluation is scalar
        _err_cdf2:           Returns squared error
        _derr_cdf2:          Derivative of squared error (require dF_dtheta to be specified at construction)
        _conf_int:           Main method used to actually return array of CIs
        """
        # Input checks
        assert callable(dist), 'dist must be callable'
        assert hasattr(dist, 'cdf'), 'dist must have a cdf method'
        assert isinstance(param_theta, str), 'param_theta needs to be a string'
        assert isinstance(alpha, float) and (alpha > 0) and (alpha < 1), 'alpha must be a float, and between 0 < alpha < 1'
        assert isinstance(verbose, bool), 'verbose must be a bool'
        assert isinstance(verbose_iter, int) and (verbose_iter > 0), 'verbose_iter must be an int > 0'
        # Assign
        self.dist = dist
        self.alpha = alpha
        self.param_theta = param_theta
        self.verbose = verbose
        self.verbose_iter = verbose_iter
        self.dF_dtheta = None
        if dF_dtheta is not None:
            assert callable(dF_dtheta), 'dF_dtheta must be callable'
            named_args = getfullargspec(dF_dtheta).args
            required_args = ['x', 'alpha', param_theta]
            assert all([n in named_args for n in required_args]), f'If the dF_dtheta function is supplied, it must have the following named arguments: {", ".join(required_args)}'
            self.dF_dtheta = dF_dtheta


    def _err_cdf(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> np.ndarray:
        """
        Calculates the differences between the candidate points and the CDF and the current set of parameters
        
        Parameters
        ----------
        theta:              Candidate parameter values to be passed into self.dist
        x:                  Points to evaluate the self.dist(...).cdf(x) at
        alpha:              Type-I error to compare the cdf to self.dist.cdf(x)-alpha
        dist_args:          If kwargs cannot be passed, assumes that the dist_args are a tuple where the first element and the key names, and the remaining are the values
        dist_kwargs:        Other named parameters which will go into self.dist({param_theta}=theta,**{'scale':2}). Note that user can also add a named key "flatten":{True,False} which will not go into the evaluation

        Returns
        -------
        A np.ndarray of the same size as theta/x
        """
        # Process the args/kwargs, and determine if need to flatten
        flatten, dist_kwargs = _process_args_kwargs_flatten(dist_args, dist_kwargs)
        # Combine the named parameter with any other ones
        dist_kwargs[self.param_theta] = theta
        # Evaluate the error
        dist = self.dist(**dist_kwargs)
        cdf = dist.cdf(x)
        err = cdf - alpha
        if flatten:
            err = err.flatten()
        return err


    def _err_cdf0(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> float:
        """Wrapper for _err_cdd to return a float"""
        res = float(self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs))
        self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs)
        return res


    def _err_cdf2(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> np.ndarray:
        """Wrapper for _err_cdf to return log(squared value + 1)"""
        err = self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs)
        err2 = np.log(np.sum(err**2) + 1)  # More stable than: np.sum(err**2)
        return err2


    def _derr_cdf2(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> np.ndarray:
        """Wrapper for the derivative of d/dmu (F(theta) - alpha)**2 = 2*(F(theta)-alpha)*(d/dmu F(mu))"""
        flatten, dist_kwargs = _check_flatten(**dist_kwargs)
        term1 = self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs)
        term2 = self.dF_dtheta(theta, x, alpha, *dist_args, **dist_kwargs)
        # res = 2 * term1 * term2
        res = (2 * term1 * term2) / ( term1**2 + 1)
        if flatten:
            res = res.flatten()
        return res

    @staticmethod
    def _process_fun_x0_x1(x:float | np.ndarray, fun_x0 : Optional[Callable] = None, fun_x1:Optional[Callable] = None) -> tuple | np.ndarray | None:
        """
        If we have a vector/float of observation values (x), and the user passes a fun_x{01} which maps x to some starting point(s), then we return those
        """
        x0, x1 = None, None
        if fun_x0 is not None:
            x0 = fun_x0(x)
        if fun_x1 is not None:
            x1 = fun_x1(x)
        if x0 is not None and x1 is not None:
            return x0, x1
        else:
            return x0

    def try_root(self, di:dict, tol:float=1e-3) -> np.ndarray:
        """
        Will try the root optimizer, and experiment with different starting values if the optimizer fails. Assumes di has been constructed so that root(**di) will run
        """
        res = root(**di)  # Run optimizer
        aerr = np.abs(res['fun'])  # Absolute value of the error
        theta = res.x  # Extract solition
        idx_fail = aerr > tol  # Check run
        theta_fail = theta[idx_fail].copy()
        n_fail = idx_fail.sum()
        if n_fail > 0:
            theta[idx_fail] = np.nan
            warn(f'{n_fail} of {len(theta)} failed, trying bounded search')
            # (i) Subset the dictionary to the failed indices
            di_fail = di.copy()
            di_fail['f'] = di_fail.pop('fun')
            del di_fail['jac']
            di_fail['x0'] = di_fail['x0'][idx_fail]
            di_fail['args'] = list(di_fail['args'])  # Only lists support index overwriting
            di_fail['args'][0] = di_fail['args'][0][idx_fail]  # These are the "x" points to solve for
            # Index 2 are the named arguments
            if 'cdf_approach' in di_fail['args'][2]:
                idx_approach = np.argmax(np.array(di_fail['args'][2]) == 'cdf_approach')
                # scipy is slower, but more stable
                di_fail['args'][3][idx_approach] = 'scipy'
            # Index 3 are all the other distribution parameters (e.g. mu, tau21)
            di_fail['args'][3] = [v[idx_fail] if isinstance(v, np.ndarray) else v for v in di_fail['args'][3]]
            di_fail['args'] = tuple(di_fail['args'])
            
            # (ii) Do a function line search and check that there is a sign flip
            sigma1_fail = np.sqrt(di_fail['args'][3][1] + di_fail['args'][3][2])
            di_fail['x0']
            x0_check = np.outer(np.arange(-10, 11, 1), sigma1_fail)
            fun_check = np.zeros(x0_check.shape)
            for k in range(len(x0_check)):
                fun_check[k] = di_fail['f'](x0_check[k], *di_fail['args'])
            # Find the first point of transition
            idx_flip = np.diff(fun_check > 0, axis=0)
            assert np.all(np.sum(idx_flip, axis=0) == 1), 'Multipe sign changes found!!'
            idx_flip = np.argmax(idx_flip, axis=0)
            x_low = x0_check[idx_flip, range(n_fail)]
            x_high = x0_check[idx_flip+1, range(n_fail)]
            # Ensure the signs do not align
            sign_low = np.sign(di_fail['f'](x_low, *di_fail['args']))
            sign_high = np.sign(di_fail['f'](x_high, *di_fail['args']))
            assert np.all(sign_low != sign_high), 'woops signs are aligned!'

            # (iii) Loop over each parameter and use the bracket
            theta_recover = np.zeros(theta_fail.shape)
            di_fail_j = di_fail.copy()
            di_fail_j['method'] = 'bisect'
            del di_fail_j['x0']
            di_fail_j['args'] = list(di_fail_j['args'])
            for j in range(n_fail):
                di_fail_j['args'][0] = di_fail['args'][0][[j]]
                di_fail_j['args'][3] = [v[[j]] if isinstance(v, np.ndarray) else v for v in di_fail['args'][3]]
                di_fail_j['bracket'] = [x_low[j], x_high[j]]
                di_fail_j['args'] = tuple(di_fail_j['args'])
                sol_j = root_scalar(**di_fail_j)
                assert sol_j.converged, 'solution j did not converge'
                theta_recover[j] = sol_j.root
                di_fail_j['args'] = list(di_fail_j['args'])
            # Update the failed vector
            theta[idx_fail] = theta_recover
        return theta


    def _conf_int(self, x:np.ndarray, approach:str='root', di_dist_args:dict | None=None, di_scipy:dict | None=None, mu_lb:float | int=-100000, mu_ub:float | int=100000, fun_x0:Optional[Callable] = None, fun_x1:Optional[Callable] = None, fun_x01_type:str='nudge', x0:np.ndarray | None=None, x1:np.ndarray | None=None) -> np.ndarray:
        """
        Parameters
        ----------
        x:                  An array-like object of points that corresponds to dimensions of estimated means
        approach:           Which scipy method to use (see scipy.optimize.{root, minimize_scalar, minimize, root_scalar}), default='root'
        di_scipy:           Dictionary to be passed into scipy optimization (e.g. root(**di_scipy), di_scipy={'method':'secant'}), default=None
        di_dist_args:       A dictionary that contains the named paramaters which are fixed for CDF calculation (e.g. {'scale':2}), default=None
        mu_lb:              Found bounded optimization methods, what is the lower-bound of means that will be considered for the lower-bound CI? (default=-100000)
        mu_ub:              Found bounded optimization methods, what is the upper-bound of means that will be considered for the lower-bound CI? (default=+100000)
        fun_x0:             Is there a function that should map x to a starting vector/float of x0?
        fun_x1:             Is there a function that should map x to a starting float of x1 (see root_scalar)?
        x0:                 For the root method, if specified gives the starting point for the lowerbound solution (default=None)
        x1:                 For the root method, if specified gives the starting point for the upperbound solution (default=None)
        
        fun_x01_type
        ------------
        nudge:          x0 -> x0, x1 -> 1.01*x1

        Optimal approaches
        ------------------
        approach            method          notes
        root_scalar         secant          More robust, slightly less accurate/speed
        root_scalar         newton          A bit slower than root_scalar, but can be more accurate, requires a_min/a_max tuning
        root                hybr            Faster, more accurate, but can blow up sometimes


        Returns
        -------
        An ({x.shape},2) array for the lower/upper bound. Shape may be different than x if x gets broadcasted by the di_dist_args
        """
        # Input checks
        assert approach in valid_approaches, f'approach needs to be one of {", ".join(valid_approaches)}'
        # Convert None to dicts
        di_dist_args = {} if di_dist_args is None else di_dist_args
        di_scipy = {} if di_scipy is None else di_scipy
        assert isinstance(di_dist_args, dict), 'if di_dist_args is not None, it needs to be a dict'
        assert isinstance(di_scipy, dict), 'if di_scipy is not None, it needs to be a dict'
    
        # Get x-initiatization mapping functions
        base_fun_x0, base_fun_x1 = _return_x01_funs(fun_x01_type)
        if fun_x0 is None:
            fun_x0 = base_fun_x0
        if fun_x1 is None:
            fun_x1 = base_fun_x1                
        
        # Broadcast x to match underlying parameters (or vice versa)
        x = np.squeeze(x)  # Squeeze x if we can in case parameters are (n,)
        tmp = broastcast_max_shape(x, *di_dist_args.values())
        x, di_dist_args = tmp[0], dict(zip(di_dist_args.keys(),tmp[1:]))
        
        # Set a default "method" for each if not supplied in the di_scipy
        if 'method' not in di_scipy:
            di_scipy['method'] = di_default_methods[approach][0]
        else:
            assert di_scipy['method'] in di_default_methods[approach], f'If method is provided to di_scipy, it must be one of: {di_default_methods[approach]}'

        # Set up the argument for the vectorized vs scalar methods
        if approach in ['minimize_scalar','root_scalar']:
            # Will be assigned iteratively
            ci_lb = np.zeros(x.shape) * np.nan
            ci_ub = ci_lb.copy()
            # Invert the di_dist_args so that the keys are the broadcasted index
            u_lens = set([v.shape[0] for v in di_dist_args.values()])
            assert len(u_lens) == 1, 'Multiple lengths found for di_dist_args, please make sure they are broadcastable to the same length'
            new_keys = range(list(u_lens)[0])
            di_dist_args_idx = {i: tuple([np.atleast_1d(di_dist_args[key][j]) for key in di_dist_args.keys()]) for i, j in enumerate(np.ndindex(x.shape))}
            # di_dist_args_idx = {i: tuple([di_dist_args[key][i] for key in di_dist_args.keys()]) for i in new_keys}
        else:
            # Needs to be flat for vector-base solvers
            assert approach in ['root','minimize'], 'expected the vector solvers'
            if approach == 'root':
                should_flatten = True
            else:
                should_flatten = False
            di_dist_args = {k:v.flatten() for k,v in di_dist_args.items()}
            arg_names = list(di_dist_args.keys()) + ['flatten']
            arg_vals = list(di_dist_args.values()) + [should_flatten]
            arg_vec = [x.flatten(), None, arg_names, arg_vals]
            

        # ---- Approach #1: Point-wise root finding ---- #
            # There are four different approaches to configure root_scalar
        if approach == 'root_scalar':
            if di_scipy['method'] in ['bisect', 'brentq', 'brenth', 'ridder','toms748']:
                di_scipy['bracket'] = (mu_lb, mu_ub)
            elif di_scipy['method'] == 'newton':
                di_scipy['x0'] = None
                assert self.dF_dtheta is not None, 'if method=="newton" is chosen, then dF_dtheta must be specified at construction'
                di_scipy['fprime'] = self.dF_dtheta
            elif di_scipy['method'] == 'secant':
                di_scipy['x0'] = mu_lb
                di_scipy['x1'] = mu_ub
            
            # Prepare the parts of the optimization that won't change (note that di_scipy will overwrite di_base)
            di_base = {'f':self._err_cdf0, 'xtol':1e-4, 'maxiter':250, 'args':(), 'x0':mu_lb, 'x1':mu_ub, 'bracket':(mu_lb, mu_ub)}  
            di_base = {**di_base, **di_scipy}  # Hard-coding unless specified by user based on stability experiments
            
            # Loop over all element points
            i = 0
            stime = time()
            n_iter = int(np.prod(x.shape))
            for kk in np.ndindex(x.shape): 
                # Prepare arguments _err_cdf0(theta, x, alpha, **other_args)
                x_kk = x[kk]
                # Update initializer
                x0_kk, x1_kk = self._process_fun_x0_x1(x_kk, fun_x0, fun_x1)
                di_base['x0'], di_base['x1'] = x0_kk, x1_kk
                # Solve lowerbound
                args_ii = [x_kk, 1-self.alpha/2, di_dist_args.keys(), di_dist_args_idx[i]]
                di_base['args'] = tuple(args_ii)
                lb_kk = root_scalar(**di_base).root
                # Solve upperbound
                args_ii[1] = self.alpha/2
                di_base['args'] = tuple(args_ii)
                ub_kk = root_scalar(**di_base).root
                # Save
                ci_lb[kk] = lb_kk
                ci_ub[kk] = ub_kk
                if self.verbose:
                    is_checkpoint = (i+1) % self.verbose_iter==0
                    if is_checkpoint:
                        dtime, nleft = time() - stime, n_iter - (i+1)
                        rate = (i+1) / dtime
                        seta = nleft / rate
                        print(f'Iteration {i+1} of {n_iter} (ETA={seta/60:0.1f} minutes)')
                # Update step
                i += 1  

        elif approach == 'minimize_scalar':
            # ---- Approach #2: Point-wise gradient ---- #
            di_base = {'fun':self._err_cdf2, 'bounds':(mu_lb, mu_ub), 'args':()}
            di_base = {**di_base, **di_scipy}
            # Loop over all element points
            i = 0
            for kk in np.ndindex(x.shape): 
                # Prepare arguments _err_cdf0(theta, x, alpha, **other_args)
                x_kk = x[kk]
                # Solve lowerbound
                args_ii = [x_kk, 1-self.alpha/2, di_dist_args.keys(), di_dist_args_idx[i]]
                di_base['args'] = tuple(args_ii)
                lb_kk = minimize_scalar(**di_base).x
                # Solve upperbound
                args_ii[1] = self.alpha/2
                di_base['args'] = tuple(args_ii)
                ub_kk = minimize_scalar(**di_base).x
                # Save
                ci_lb[kk] = lb_kk
                ci_ub[kk] = ub_kk
                # Update step
                i += 1 
        
        elif approach == 'minimize':
            # ---- Approach #3: Vector gradient ---- #
            di_base = {**{'fun':self._err_cdf2, 'args':(), 'x0':fun_x0(x.flatten())}, **di_scipy}
            if di_scipy['method'] in ['CG','BFGS','L-BFGS-B','TNC','SLSQP']:
                # Gradient methods
                di_base['jac'] = self._derr_cdf2
            else:
                # Gradient free methods
                assert di_scipy['method'] in ['Nelder-Mead', 'Powell', 'COBYLA']
            # Solve for the lower-bound
            arg_vec[1] = 1-self.alpha/2
            di_base['args'] = tuple(arg_vec)
            ci_lb = minimize(**di_base).x
            # Solve for upper-bound
            arg_vec[1] = self.alpha/2
            di_base['args'] = tuple(arg_vec)
            ci_ub = minimize(**di_base).x
            # Return to original size
            ci_lb = ci_lb.reshape(x.shape)
            ci_ub = ci_ub.reshape(x.shape)

        else:
            # ---- Approach #4: Vectorized root finding ---- #
            if x0 is None:
                x0 = fun_x0(x.flatten())
            if x1 is None:
                x1 = fun_x1(x.flatten())
            di_base = {'fun':self._err_cdf, 'jac':self.dF_dtheta, 'args':()}
            di_base = {**di_base, **di_scipy}
            # Solve for the lower-bound
            arg_vec[1] = 1-self.alpha/2
            di_base['args'] = tuple(arg_vec)
            di_base['x0'] = x0  # Use x0 for the lowerbound
            ci_lb = self.try_root(di_base)  # root(**di_base).x
            # Solve for upper-bound
            arg_vec[1] = self.alpha/2
            di_base['args'] = tuple(arg_vec)
            di_base['x0'] = x1  # Use x1 for the upperbound
            ci_ub = self.try_root(di_base)
            # Return to original size
            ci_lb = ci_lb.reshape(x.shape)
            ci_ub = ci_ub.reshape(x.shape)
        if np.any(np.isnan(ci_lb) | np.isnan(ci_ub)):
            warn('Null values detected! in the confidence interval, something almost surely went wrong')
        # Check which order to return the columns so that lowerbound is in the first column position
        is_correct = np.all(ci_ub >= ci_lb)
        is_flipped = False
        if not is_correct:
            is_flipped = np.all(ci_lb >= ci_ub)
            if not is_flipped:
                warn('The upperbound is not always larger than the lowerbound! Something probably went wrong...')
        # Return values
        if is_flipped:
            mat = np.stack((ci_ub, ci_lb),ci_lb.ndim)
        else:
            mat = np.stack((ci_lb,ci_ub),ci_lb.ndim)
        return mat 