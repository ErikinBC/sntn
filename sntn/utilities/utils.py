"""
Utility functions
"""

# Load modules
import os
import numpy as np
from typing import Type, Callable
from collections.abc import Iterable

def try2list(x) -> list:
    """Try to convert x to a list"""
    is_iterable = isinstance(x, Iterable)
    if isinstance(x, list):
        return x
    elif is_iterable:
        return list(x)
    else:
        return [x]


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



