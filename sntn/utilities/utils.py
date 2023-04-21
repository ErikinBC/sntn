"""
Utility functions
"""

# Load modules
import os
import numpy as np
import pandas as pd
from mizani.transforms import trans
from collections.abc import Iterable
from typing import Type, Callable, Tuple


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
    if (a_min is None) and (a_max is None):
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


class pseudo_log10(trans):
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
            pos_expand_axis = list(n_sarg + np.arange(n_max_shape - n_sarg))
            args_i = np.expand_dims(arg, axis=pos_expand_axis)
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