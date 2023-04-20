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
from copy import deepcopy
from inspect import getfullargspec
from scipy.stats import norm
from scipy.optimize import root, minimize_scalar, minimize, root_scalar
# Internal modules
from sntn.utilities.utils import broastcast_max_shape, str2list, try2list, no_diff

# Hard-coded scipy approaches and methods
valid_approaches = ['root', 'minimize_scalar', 'minimize', 'root_scalar']
di_default_methods = {'root':'hybr',
                      'minimize_scalar':'Golden',
                      'minimize':'Powell',
                      'root_scalar':'secant'}  # Which method to use for each approach
no_diff(valid_approaches, di_default_methods.keys())



class conf_inf_solver():
    def __init__(self, dist:callable, param_theta:str, dF_dtheta:None or callable=None, alpha:float=0.05) -> None:
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
        # Assign
        self.dist = dist
        self.alpha = alpha
        self.param_theta = param_theta
        self.dF_dtheta = None
        if dF_dtheta is not None:
            assert callable(dF_dtheta), 'dF_dtheta must be callable'
            named_args = getfullargspec(dF_dtheta).args
            required_args = ['x', 'alpha', param_theta]
            assert all([n in named_args for n in required_args]), f'If the dF_dtheta function is supplied, it must have the following named arguments: {", ".join(required_args)}'
            self.dF_dtheta = dF_dtheta


    def _check_flatten(self, dist_kwargs:dict) -> bool:
        """
        Used by the _err_cdf and _err_cdf2 functions to see if there is a 'flatten' argument that gets passed into dist_kwargs. Will return the boolean value of flatten, and then remove the key from the dict so as to not cause an error for the self.dist() function
        """
        # Check if we will flatten
        flatten = False
        if 'flatten' in dist_kwargs:
            flatten = dist_kwargs['flatten']
            assert isinstance(flatten, bool), 'If flatten is added to **dist_kwargs, then it needs to a bool'
            del dist_kwargs['flatten']  # Remove  (NOTE, WE MAY NEED TO USE A DEEP COPY HERE???)
        return flatten


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
        if len(dist_args) > 0:
            di_names = dist_args[0]
            dist_kwargs = dict(zip(di_names, dist_args[1:]))
        flatten = self._check_flatten(dist_kwargs)
        # Combine the named parameter with any other ones
        kwargs = {**{self.param_theta:theta}, **dist_kwargs}
        # Evaluate the error
        err = self.dist(**kwargs).cdf(x) - alpha
        if flatten:
            err = err.flatten()
        return err


    def _err_cdf0(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> float:
        """Wrapper for _err_cdd to return a float"""
        res = float(self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs))
        return res


    def _err_cdf2(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> np.ndarray:
        """Wrapper for _err_cdf to return squared value"""
        err = self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs)
        err2 = np.sum(err**2)
        return err2


    def _derr_cdf2(self, theta:np.ndarray, x:np.ndarray, alpha:float, *dist_args, **dist_kwargs) -> np.ndarray:
        """Wrapper for the derivative of d/dmu (F(theta) - alpha)**2 = 2*(F(theta)-alpha)*(d/dmu F(mu))"""
        flatten = self._check_flatten(dist_kwargs)
        term1 = 2*self._err_cdf(theta, x, alpha, *dist_args, **dist_kwargs)
        term2 = self.dF_dtheta(theta, x, alpha, *dist_args, **dist_kwargs)
        res = term1 * term2
        if flatten:
            res = res.flatten()
        return res


    def _conf_int(self, x:np.ndarray, approach:str='root_scalar', di_dist_args:dict or None=None, di_scipy:dict or None=None, mu_lb:float or int=-100000, mu_ub:float or int=100000) -> np.ndarray:
        """
        Parameters
        ----------
        x:                  An array-like object of points that corresponds to dimensions of estimated means
        approach:           Which scipy method to use (see scipy.optimize.{root, minimize_scalar, minimize, root_scalar}), default='root_scalar'
        di_dist_args:       A dictionary that contains the named paramaters which are fixed for CDF calculation (e.g. {'scale':2}), default=None
        di_scipy:           Dictionary to be passed into scipy optimization (e.g. root(**di_scipy), di_scipy={'method':'secant'}), default=None
        mu_lb:              Found bounded optimization methods, what is the lower-bound of means that will be considered for the lower-bound CI? (default=-100000)
        mu_ub:              Found bounded optimization methods, what is the upper-bound of means that will be considered for the lower-bound CI? (default=+100000)

        Optimal approaches
        ------------------
        approach            method          notes
        root_scalar         secant          Optimal speed/accuracy
        minimize_scalar     Golden          Slower, but best accuracy


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
        
        # Broadcast x to match underlying parameters (or vice versa)
        x = np.squeeze(x)  # Squeeze x if we can in cause parameters are (n,)        
        tmp = broastcast_max_shape(x, *di_dist_args.values())
        x, di_dist_args = tmp[0], dict(zip(di_dist_args.keys(),tmp[1:]))
        
        # Set a default "method" for each if not supplied in the di_scipy
        if 'method' not in di_scipy:
            di_scipy['method'] = di_default_methods[approach]

        # Set up the argument for the vectorized vs scalar methods
        if approach in ['minimize_scalar','root_scalar']:
            # Will be assigned iteratively
            ci_lb = np.full_like(x, fill_value=np.nan)
            ci_ub = ci_lb.copy()
            # Invert the di_dist_args so that the keys are the broadbasted index
            u_lens = set([v.shape[0] for v in di_dist_args.values()])
            assert len(u_lens) == 1, 'Multiple lengths found for di_dist_args, please make sure they are broadcastable to the same length'
            new_keys = range(list(u_lens)[0])
            di_dist_args_idx = {i: tuple([di_dist_args[key][i] for key in di_dist_args.keys()]) for i in new_keys}
        else:
            # Needs to be flat for vector-base solvers
            should_flatten = True
            di_dist_args = {k:v.flatten() for k,v in di_dist_args.items()()}

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
            di_base = {'f':self._err_cdf0, 'xtol':1e-4, 'maxiter':250}  # Hard-coding unless specified by user based on stability experiments
            
            # Loop over all element points
            i = 0
            for kk in np.ndindex(x.shape): 
                # Prepare arguments _err_cdf0(theta, x, alpha, **other_args)
                x_kk = x[kk]
                args_ii = di_dist_args_idx[i]
                di_scipy['x0'] = 1.00*x_kk
                di_scipy['x1'] = 1.01*x_kk
                # Prepare optional arguments to pass into loss function
                args_ii_lb = [x_kk, 1-self.alpha/2, di_dist_args.keys(), args_ii]
                # Swap alpha's for upper/lowerbound
                args_ii_ub = args_ii_lb.copy()
                args_ii_ub[1] = self.alpha/2
                # Prepare final dict for root finding
                di_lb = {**di_base, **di_scipy}
                di_ub = {**di_base, **di_scipy}
                di_lb['args'] = tuple(args_ii_lb)
                di_ub['args'] = tuple(args_ii_ub)
                lb_kk = float(root_scalar(**di_lb).root)
                ub_kk = float(root_scalar(**di_ub).root)
                ci_lb[kk] = lb_kk
                ci_ub[kk] = ub_kk
                i += 1  

        elif approach == 'minimize_scalar':
            # ---- Approach #2: Point-wise gradient ---- #
            di_base = {'fun':self._err_cdf2, 'bounds':(mu_lb, mu_ub)}
            di_base = {**di_base, **di_scipy}
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
            di_base = {**{'fun':self._err_cdf2, 'args':solver_args, 'x0':mu.flatten()},**di_scipy}
            if di_scipy['method'] in ['CG','BFGS','L-BFGS-B','TNC','SLSQP']:
                # Gradient methods
                di_base['jac'] = self._derr_cdf2
            elif di_scipy['method'] in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
                raise Warning(f"The method you have specified ({di_scipy['method']}) is not supported")
                return None
            else:
                # Gradient free methods
                assert di_scipy['method'] in ['Nelder-Mead', 'Powell', 'COBYLA']
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
            # Guess some lower/upper bounds
            c_alpha = norm.ppf(self.alpha/2)
            x0_lb, x0_ub = self.mu + c_alpha, self.mu - c_alpha
            di_base = {**{'fun':self._err_cdf, 'jac':self._dmu_dcdf, 'args':solver_args},**di_scipy}
            if di_scipy['method'] in ['broyden1', 'broyden2', 'anderson', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']:
                raise Warning(f"The method you have specified ({di_scipy['method']}) is not supported")
                del di_base['jac']
            else:
                assert di_scipy['method'] in ['hybr', 'lm']
            di_lb, di_ub = deepcopy(di_base), deepcopy(di_base)
            di_lb['args'][4] = 1-alpha/2
            di_lb['x0'] = x0_lb.flatten()
            di_ub['args'][4] = alpha/2
            di_ub['x0'] = x0_ub.flatten()
            di_lb['args'], di_ub['args'] = tuple(di_lb['args']), tuple(di_ub['args'])
            ci_lb = root(**di_lb).x
            ci_ub = root(**di_ub).x
        # Check which order to return the columns so that lowerbound is in the first column position
        is_correct = np.all(ci_ub >= ci_lb)
        is_flipped = False
        if not is_correct:
            is_flipped = np.all(ci_lb >= ci_ub)
            if not is_flipped:
                raise Warning('The upperbound is not always larger than the lowerbound! Something probably went wrong...')
        # Return values
        if is_flipped:
            mat = np.stack((ci_ub, ci_lb),ci_lb.ndim)
        else:
            mat = np.stack((ci_lb,ci_ub),ci_lb.ndim)
        return mat 