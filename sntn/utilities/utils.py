"""
Utility functions
"""

# Load modules
import os
import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import Type, Callable, Tuple
from inspect import getfullargspec, signature

# Return lb/ub confidence intervals on a binomial proportion
def get_CI(df:pd.DataFrame, cn_den:str, cn_pct:None or str=None, cn_num:None or str=None, alpha:float=0.05, method='beta') -> pd.DataFrame:
    """
    Add on the binomial proportion CI to an existing DataFrame. User must specify either the percent of successes (cn_pct) or number of successes (cn_num)

    Params
    ======
    cn_den:             Column with the number of observations (denominator)
    cn_pct:             Column with the percent of successes
    cn_num:             Column giving the number of successes (numerator)
    alpha:              Type-I error rate
    method:             See proportion_confint(method=...)
    """
    from statsmodels.stats.proportion import proportion_confint as prop_CI
    assert (cn_pct is not None) or (cn_num is not None), 'At least one of cn_{pct,num} must be specified'
    num_tries = df[cn_den].copy()
    if cn_pct is not None:  # count can be unknown
        assert isinstance(cn_pct, str), 'if cn_pct is not None, then it must be a string'
        pct_success = df[cn_pct].copy()
        num_successes = pct_success * num_tries
    else:  # percent can be unknwoen
        assert isinstance(cn_num, str), 'if cn_prop is None, then cn_count needs to be a string'
        num_successes = df[cn_num].copy()
        pct_success = num_successes / num_tries
    tmp_CI = prop_CI(count=num_successes, nobs=num_tries, alpha=alpha, method=method)
    tmp_CI = pd.DataFrame(np.vstack(tmp_CI).T, columns=['lb','ub'])
    res = pd.concat(objs=[df, tmp_CI], axis=1)
    assert np.all( (res['lb'] <= pct_success) & (res['ub'] >= pct_success) ), 'Woops! [lb,ub] do not bracket the percent of successes!'
    return res


def try_except_breakpoint(cond:bool, stmt:str or None=None):
    if stmt is None:
        stmt = ''
    try:
        assert cond, stmt
    except:
        breakpoint()
        cond


def get_valid_kwargs_cls(cls, **kwargs):
    # Get the class constructor's parameter names
    sig = signature(cls.__init__)
    valid_kwargs = set(sig.parameters.keys()) - {'self'}    
    # Filter the input kwargs to only include valid kwargs
    return {k: v for k, v in kwargs.items() if k in valid_kwargs}


def get_valid_kwargs_method(obj, method_name, **kwargs):
    method = getattr(obj, method_name)
    arg_names = getfullargspec(method).args[1:]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in arg_names}
    return filtered_kwargs


def pass_kwargs_to_classes(cls, *args, **kwargs):
    """Try passing keywords to class, and the class will only use those kwargs that match named arguments"""
    arg_names = set(getfullargspec(cls.__init__).args[1:])
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in arg_names}
    return cls(*args, **filtered_kwargs)



def process_x_x1_x2(x:np.ndarray or None=None, x1:np.ndarray or None=None, x2:np.ndarray or None=None) -> tuple:
    """
    For the bivariate normal, extract the two coordinates

    Args:
    ----
    x (np.ndarray):
        A (n1,..,nj,d1,..,dk,2) array
    x1 (np.ndarray):
        If x is not specified...
    x2 (np.ndarray):
        Second coordinate if x not specified

    Returns
    -------
    An tuple of (d1,..,dk) arrays
    """
    if x is None:
        assert x1 is not None and x2 is not None, 'if x is not specified, then x1/x2 need to be'
        x1, x2 = np.broadcast_arrays(x1, x2)
    else:
        x = np.asarray(x)
        assert x.shape[-1] == 2, 'if x is given (rather than x1/x2), last dimension needs to be of size 2'
        # Take out x1/x2
        x1, x2 = np.take(x, 0, -1), np.take(x, 1, -1)
    return x1, x2


def array_to_dataframe(arr:np.ndarray) -> pd.DataFrame:
    """Converts an (n, d1, ..., dk) array into a DataFrame of dimension (n, d1*...*dk) with multindexing for the columns"""
    shape = arr.shape
    n = shape[0]
    dims = shape[1:]
    index = pd.MultiIndex.from_product([range(d) for d in dims], names=[f'd{i}' for i in range(1, len(dims)+1)])
    values = arr.reshape(n, -1)
    return pd.DataFrame(values, columns=index)


def try2array(x) -> np.ndarray:
    """Checks to see if x is ndarray, and if not tries to make at least-1d"""
    if not isinstance(x, np.ndarray):
        return np.atleast_1d(x)
    else:
        return x


def no_diff(x, y):
    assert set(x) == set(y) and len(x) == len(y), 'x and y do not have the same elements'


def try2list(x) -> list or tuple:
    """If x is not a list or a tuple, return as a list"""
    if not (isinstance(x, list) or isinstance(x, tuple)):
        return [x]
    else:
        return x


def str2list(x:str or list) -> list:
    """
    If x is a string, convert to a list
    """
    if isinstance(x, str):
        return [x]
    else:
        return x


def cat_by_order(df:pd.DataFrame, cn_order:str or list, cn_cat:str, ascending:bool or list=True, drop_index:bool=True) -> pd.DataFrame:
    """
    Sort a dataframe and have one of the column become a category based on that order

    Parameters
    ----------
    df:                 DataFrame to be sorted
    cn_order:           Names of columns to sort by
    cn_cat:             Name of column to make into categorical (must be unique)
    ascending:          To be passed into pd.DataFrame.sort_values (default=True)
    drop_index:         Whether index should be dropped after sort (default=True)

    Returns
    -------
    DataFrame of the same shape
    """
    # Input checks
    cn_order = str2list(cn_order)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.isin(cn_order).sum() == len(cn_order), 'cn_order not found in df'
    assert df.columns.isin(str2list(cn_cat)).sum() == 1, 'cn_cat not found in df'
    df = df.sort_values(cn_order, ascending=ascending)
    if drop_index:
        df = df.reset_index(drop=True)
    df[cn_cat] = pd.Categorical(df[cn_cat], df[cn_cat].values)
    return df



def mean_abs_error(x:np.ndarray) -> float:
    """Assuming x is the error"""
    return np.mean(np.abs(x))

def mean_total_error(x:np.ndarray) -> float:
    """Assuming x is the error"""
    return np.sum(np.abs(x))


def grad_clip_abs(x:np.ndarray, a_min:float or None=None, a_max:float or None=None) -> np.ndarray:
    """Return the absolute value of a gradient value either rounded up or down (a_min/a_max should be positive)"""
    if isinstance(a_min, np.ndarray) and isinstance(a_max, np.ndarray):
        if None in a_min and None in a_max:
            return x
    elif (a_min is None) and (a_max is None):
        return x
    sx = np.sign(x)
    cx = sx * np.clip(np.abs(x), a_min, a_max)
    return cx


def is_equal(x:np.ndarray, y:np.ndarray, tol:float=1e-10) -> None:
    """Can an assertion error if max(abs(x-y)) >= tol"""
    mx_err = np.max(np.abs(x-y))
    assert np.all( mx_err <= tol ), f'Error! Maximum error {mx_err} is greater than tolerance {tol}'


def try2list(x) -> list:
    """Try to convert x to a list"""
    is_iterable = isinstance(x, Iterable)
    if isinstance(x, list):
        return x
    elif is_iterable:
        return list(x)
    else:
        return [x]


def pn_labeller(rows=None, cols=None, multi_line=True, **kwargs):
    """
    Return a labeller function

    Parameters
    ----------
    rows : str | function | None
        How to label the rows
    cols : str | function | None
        How to label the columns
    multi_line : bool
        Whether to place each variable on a separate line
    default : function | str
        Fallback labelling function. If it is a string,
        it should be the name of one the labelling
        functions provided by plotnine.
    kwargs : dict
        {variable name : function | string} pairs for
        renaming variables. A function to rename the variable
        or a string name.

    Returns
    -------
    out : function
        Function to do the labelling
    """
    from plotnine import as_labeller, label_value
    # Sort out the labellers along each dimension
    rows_labeller = as_labeller(rows, label_value, multi_line)
    cols_labeller = as_labeller(cols, label_value, multi_line)

    def collapse_label_lines(label_info):
        """
        Concatenate all items in series into one item
        """
        return pd.Series([', '.join(label_info)])


    def _labeller(label_info):
        # When there is no variable specific labeller,
        # use that of the dimension
        if label_info._meta['dimension'] == 'rows':
            margin_labeller = rows_labeller
        else:
            margin_labeller = cols_labeller

        # Labelling functions expect string values
        label_info = label_info.astype(str)

        # Each facetting variable is labelled independently
        for name, value in label_info.items():
            func = as_labeller(kwargs.get(name), margin_labeller)
            new_info = func(label_info[[name]])
            label_info[name] = new_info[name]

        if not multi_line:
            label_info = collapse_label_lines(label_info)

        return label_info

    return _labeller


class pseudo_log10():
    """
    Implements the pseudo-log10 transform: https://win-vector.com/2012/03/01/modeling-trick-the-signed-pseudo-logarithm/
    """
    @staticmethod
    def transform(x):
        # y = arcsinh(x/2) / log(10)
        return np.arcsinh(x/2)/np.log(10)

    @staticmethod
    def inverse(x):
        # x = y*log(10)
        return np.sinh(x * np.log(10))


def makeifnot(dir_path):
    """Make a directory if it does not exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)    


def cvec(x:np.ndarray) -> np.ndarray:
    """Convert x into a column vector"""
    if len(x.shape) <= 1:
        return x.reshape([len(x),1])
    else:
        return x


def rvec(x:np.ndarray) -> np.ndarray:
    """Convert x into a row vector"""
    if len(x.shape) <= 1:
        return x.reshape([1, len(x)])
    else:
        return x


def vprint(stmt, verbose=True):
    """Print if verbose==True"""
    if verbose:
        print(stmt)


def check_all_is_type(*args, dtype:Type or list):
    """Checks that all *args match one type"""
    dtypes = try2list(dtype)
    assert all([isinstance(d, Type) for d in dtypes]), 'dtype(s) need be a type'
    if len(args) > 0:
        for a in args:
            assert any([isinstance(a, d) for d in dtypes]), f'{a} is not of type {dtype}'


def has_assertion_error(function:Callable, *args, **kwargs) -> bool:
    """See if a function returns an assertion error (if so, returns True)"""
    assert isinstance(function, Callable), 'function needs to be callabe'
    try:
        function(*args, **kwargs)
        has_error = False
    except:
        has_error = True
    return has_error


def check_all_pos(*args, strict:bool=False) -> None:
    """Checks that a list of ints or float is strictly positive"""
    lb = 0
    if strict:
        lb += 1e-16
    if len(args) > 0:
        for a in args:
            assert a >= lb, f'{a} needs to be >= {lb}'


def get_max_shape(*args) -> list:
    """For an arbitrary number of inputs, will find those that has a shape attribute and determine the largest shape (i.e. [10,2] so that we can call x.reshape([10,2]) for example)"""
    assert len(args) > 0, 'Expected at least one *args'
    # Add the shape of each arg
    holder = []
    for a in args:
        if hasattr(a, 'shape'):
            a_s = a.shape
        else:
            a_s = np.atleast_1d(a).shape
        holder.append(a_s)
    # Get the length (i.e. diension) of each
    lholder = [len(h) for h in holder]
    mlholder = max(lholder)
    holder = [h for l,h in zip(lholder,holder) if l == mlholder]
    shape_star = holder[np.argmax([np.prod(h) for h in holder])]
    return shape_star


def broastcast_max_shape(*args, **kwargs) -> Tuple:
    """
    Takes an arbitrary number of *args in, and will try to broadcast into the max shape. Will return list of the same length. For example is args=(1, [1,2,3]) then will return (array(1,2,3), array(1,2,3)).
    """
    # --- Input checks --- #
    assert len(args) > 0 or len(kwargs) > 0, 'Please specify at least one *args'
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        del kwargs['verbose']
    else:
        verbose = False
    assert isinstance(verbose, bool), 'verbose needs to be a boolean if specified'

    # --- Merge args/kwargs together --- #
    if len(kwargs) > 0:
        args += tuple(kwargs.values())
    
    # --- Ensure minimum dim --- #
    args = [np.atleast_1d(a) for a in args]

    # --- Modify each element --- #
    # Determine max shape 
    max_shape = get_max_shape(*args)
    # get the # of dimensions (n), the largest dimension (d), and total # of data points (t)
    n_max_shape, t_shape = len(max_shape), int(np.prod(max_shape))
    assert isinstance(max_shape, tuple), 'Thought get_max_shape would return tuple'
    assert n_max_shape >= 1, 'Expected np.atleast_1d to guarantee at least one dimension'
    # Ensure everything is at least 1-d (and an array) and args is a list
    args = [np.atleast_1d(arg) for arg in args]
    # Loop over each argument and attempt to broadcast
    for i, arg in enumerate(args):
        # Get arg specific dimenions
        sarg = arg.shape
        n_sarg, d_arg = len(sarg), max(sarg)
        if sarg == max_shape:
            vprint('# --- Case 1: Already matches existing shape --- #', verbose)
            continue
        # --- Case 2: As a singleton --- #
        if n_sarg == 1 and d_arg == 1:
            vprint('# --- Case 2: As a (1,) float/int --- #', verbose)
            args_i = np.repeat(arg, t_shape).reshape(max_shape) # Simply repeat and broadcast
        # --- Case 3: Same dim, different values --- #
        elif n_max_shape == n_sarg:
            vprint(f'# --- Case 3: shape lens match --- #', verbose)            
            # Broadcast any dimension with a value of one
            idx_dim1 = np.where(np.array(sarg) == 1)[0]
            assert len(idx_dim1) > 0, f'Since {sarg} does not match {max_shape}, it must have at least one dimension equal to one for broadcasting'
            tile_dim = [m if s == 1 else 1 for m, s in zip(max_shape, sarg)]
            args_i = np.tile(arg, reps=tile_dim)
        else:
            vprint(f'# --- Case 4: different dimensions --- #', verbose)
            assert n_max_shape > n_sarg, f'If we are in case 4, expected {n_max_shape} > {n_sarg}'
            # Else, we assume that we only need to append a one (if this fails it will be caught at the assertion error below)
            args_i = np.broadcast_to(arg, np.broadcast_shapes(max_shape, sarg))
            # pos_expand_axis = list(n_sarg + np.arange(n_max_shape - n_sarg))
            # args_i = np.expand_dims(arg, axis=pos_expand_axis)
        # Check and then update
        assert args_i.shape == max_shape, f'Woops expected {args_i.shape} == {max_shape}'
        args[i] = args_i
    
    # Return broadcasted args
    return args


def check_err_cdf_tol(solver, theta:np.ndarray, x:np.ndarray, alpha:float, **dist_kwargs) -> None:
    """Make sure that the root/squared error is zero and the correct solution
    
    Parameters
    ----------
    solver:             The constructed conf_inf_solver class
    theta:              Candidate CI value (i.e. upper or lower bound)
    x:                  Observed statistic
    alpha:              Type-I error (i.e. alpha/2)
    dist_kwargs:        Arguments to pass into solver._{err_cdf,err_cdf0,err_cdf2,derr_cdf2}
    """
    has_dF_dtheta = hasattr(solver, 'dF_dtheta')
    n = len(x)
    nudge = 0.1
    # Going to broadcast alpha for looping
    _, alpha = np.broadcast_arrays(theta, alpha)
    # Check that the "error" is zero at the true solution
    di_eval = {**{'theta':theta.copy(), 'x':x, 'alpha':alpha}, **dist_kwargs}
    assert np.all(solver._err_cdf(**di_eval) == 0), 'Root was not zero'
    assert np.all(solver._err_cdf2(**di_eval) == 0), 'Squared-error was not zero'
    for i in range(n):
        di_eval_i = {k:v[i] for k,v in di_eval.items()}
        assert solver._err_cdf0(**di_eval_i) == 0, 'Root (float) was not zero'
    if has_dF_dtheta:
        assert np.all(solver._derr_cdf2(**di_eval) == 0), 'Derivative was not zero'
    # Check the error is non-zero at a permuted distribution
    di_eval['theta'] += nudge
    assert np.all(solver._err_cdf(**di_eval) != 0), 'Root was zero'
    assert np.all(solver._err_cdf2(**di_eval) != 0), 'Squared-error was zero'
    if has_dF_dtheta:
        assert np.all(solver._derr_cdf2(**di_eval) != 0), 'Derivative was zero'


def flip_last_axis(x:np.ndarray) -> np.ndarray:
    """Flips a np.array so the last axis becomes first"""
    return x.transpose((len(x.shape)-1,) + tuple(range(len(x.shape)-1)))


def fast_corr(A:np.ndarray, B:np.ndarray) -> np.ndarray:
    """Calculates the column-wise correlations between two arrays"""
    # calculate the means of each column of A and B
    mean_A = np.mean(A, axis=0)
    mean_B = np.mean(B, axis=0)
    # subtract the means from each column of A and B
    A_centered = A - mean_A
    B_centered = B - mean_B
    # calculate the norm of each column of A and B
    norm_A = np.linalg.norm(A_centered, axis=0)
    norm_B = np.linalg.norm(B_centered, axis=0)
    # calculate the dot product of each pair of centered columns in A and B
    dot_products = np.sum(A_centered * B_centered, 0)
    # divide each dot product by the product of the corresponding norms to get the correlation coefficient
    correlations = dot_products / (norm_A * norm_B)
    return correlations


def rho_debiased(x:np.ndarray, y:np.ndarray, method:str='fisher') -> np.ndarray:
    """
    Return a debiased version of the emprical correlation coefficient
    
    Parameters
    ----------
    x:              A (n,i,j,k,...) dim array
    y:              Matches dim of x

    Returns
    -------
    An (i,j,k,...) array of correlation coefficients
    """
    valid_methods = ['pearson', 'fisher', 'olkin']
    assert method in valid_methods, f'method must be one of {valid_methods}'
    assert x.shape == y.shape, 'x and y must be the same shapes'
    # Assume the first dimension is size
    n = x.shape[0]
    rho = fast_corr(x, y)
    if method == 'pearson':
        return rho
    if method == 'fisher':
        return rho * (1 + (1-rho**2)/(2*n))
    if method == 'olkin':
        return rho * (1 + (1-rho**2)/(2*(n-3)))



def broadcast_to_k(x:np.ndarray, param_shape:tuple) -> np.ndarray:
    """When an array is passed into the a pdf/cdf/ppf method, we need to determine how to broadbast the array for doing inference. Suppose k is the flattened dimension of some input parameters, then when the input is a matrix, 
    
    nd = len(x.shape)
    if nd == 1:
        if x.shape[0] == k:
            Assume we want value returned for that specific parameter. For example, if the input parameters are (10,) and x is (10,) then it will be a 1:1 index mapping (index 1 == parameter 1 == pdf at parameter 1)
        else:
            Assume that we want to broadcast it. For example, if input parameters are (10,) and x is (9,), assume we want to calculate the points for all 10 parameters (i.e. 90 calculations)
    else:
        When x has 2 or more dimensions, the its input dimensions have to match the input parameters for the last k. For example, if the input parameters are (4,3,2) then x can be (6,6,4,3,2) but not (6,6,3,2) since (...,4,3,2) need to match

    Parameters
    ----------
    x:                      An array or arbitray dimension (the last d dimensions need to match len(param_shape) unless param shape is an array in which case it will an outer sweep)
    param_shape:            The dimension that we want to broadcast for
    """
    # Input checks
    assert isinstance(x, np.ndarray), 'x needs to be an array'
    assert isinstance(param_shape, tuple)
    k = int(np.prod(param_shape))
    # Go through difference options
    nd_x = len(x.shape)
    if nd_x == 1:
        if x.shape[0] == k:
            # Can return as is (e.g. mu - x) will align in dimension
            return x
        else:
            # Assume outer product type calculation
            return np.tile(x, [k,1]).T
    else:
        # The last dimensions need to match
        nd_param = len(param_shape)
        assert nd_x >= nd_param, f'if x is 2 or more dimensions, the last {nd_param} must match the param_shape={param_shape}'
        nd_x_last = x.shape[-nd_param:]
        assert nd_x_last == param_shape, f'The last {nd_param} must match the param_shape={param_shape}: {nd_x_last}'
        # If the dimensions match, then we can flatten the last dimensions
        return x.reshape(x.shape[:-nd_param] + (k,))


def reverse_broadcast_from_k(x: np.ndarray, param_shape:tuple, suffix_shape:tuple or None=None) -> np.ndarray:
    """Reverse the broadcasting performed by the broadcast_to_k function.

    Parameters
    ----------
    x : np.ndarray
        The array to reverse the broadcasting for.
    param_shape : tuple
        The original param_shape passed to broadcast_to_k.
    suffix_shape : tuple or None
        An extra shape value found at the end of x (default=None)

    Returns
    -------
    np.ndarray
        The array with the original shape before broadcasting.
    """
    # Input checks
    assert isinstance(x, np.ndarray), 'x needs to be an array'
    assert isinstance(param_shape, tuple)
    if suffix_shape is None:
        suffix_shape = ()
    else:
        assert isinstance(suffix_shape, tuple), 'if suffix_shape is not None, it needs to be a tuple'
    n_dim_suffix = len(suffix_shape)
    n_suffix = int(np.prod(suffix_shape))
    k = int(np.prod(param_shape) * n_suffix)
    n_param_shape = len(param_shape)
    x_shape = x.shape
    # How many of the last dimensions to check
    n_last_dim = 1 + n_dim_suffix
    if len(x_shape) == 1:
        assert np.prod(x_shape) == np.prod(param_shape), f'Expected number of dimensions to align'
        x = x.reshape(param_shape)
        # assert x_shape == param_shape, f'If x is a vector, expect it to match original shape {str(x_shape)}!={param_shape} (e.g. 8 == 8)'
        
    elif len(param_shape) == 1:
        assert np.prod(x_shape[-n_last_dim:]) == k, 'When parameter is a scalar, expecting last dimensions to be equal to k (e.g. (4,3,10) == (10))'
    else:
        assert np.prod(x.shape[-n_last_dim:]) == k, f"Dimensions {n_param_shape} onwards should be of size k={k}"
        # Reshape the array to the original shape
        original_shape = (x.shape[0],) + param_shape + suffix_shape
        x = x.reshape(original_shape)
    return x
