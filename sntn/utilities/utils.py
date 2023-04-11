"""
Utility functions
"""

# Load modules
import os
import numpy as np
from mizani.transforms import trans
from collections.abc import Iterable
from typing import Type, Callable, Tuple


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



def broastcast_max_shape(*args, verbose=False) -> Tuple:
    """
    Takes an arbitrary number of *args in, and will try to broadcast into the max shape. Will return list of the same length. For example is args=(1, [1,2,3]) then will return (array(1,2,3), array(1,2,3)).
    """
    # Input checks
    assert len(args) > 0, 'Please specify at least one *args'
    # assert not any([isinstance(a,list) or isinstance(a, tuple) for a in args]), 'One of the *args was found to be a list or a tuple, please make sure you specify *args if args is a list'
    
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
            print(f'# --- Case 3: shape lens match --- #', verbose)            
            # Broadcast any dimension with a value of one
            idx_dim1 = np.where(np.array(sarg) == 1)[0]
            assert len(idx_dim1) > 0, f'Since {sarg} does not match {max_shape}, it must have at least one dimension equal to one for broadcasting'
            tile_dim = [m if s == 1 else 1 for m, s in zip(max_shape, sarg)]
            args_i = np.tile(arg, reps=tile_dim)
        else:
            print(f'# --- Case 4: different dimensions --- #', verbose)
            assert n_max_shape > n_sarg, f'If we are in case 4, expected {n_max_shape} > {n_sarg}'
            # Else, we assume that we only need to append a one (if this fails it will be caught at the assertion error below)
            pos_expand_axis = list(n_sarg + np.arange(n_max_shape - n_sarg))
            args_i = np.expand_dims(arg, axis=pos_expand_axis)
        # Check and then update
        assert args_i.shape == max_shape, f'Woops expected {args_i.shape} == {max_shape}'
        args[i] = args_i
    # Return broadcasted args
    return args