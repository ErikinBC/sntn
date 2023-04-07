"""
Test utilities functions
------------------------
python3 -m tests.test_utils
python3 -m pytest tests/test_utils.py
"""

import numpy as np
import pandas as pd
from sntn.utilities.utils import check_all_is_type, try2list, has_assertion_error, check_all_pos, get_max_shape, broastcast_max_shape

def test_try2list() -> None:
    """Check that output is list"""
    x = [1,2,3]
    assert try2list(x) == x
    assert try2list(np.array(x)) == x
    assert all([try2list(z) == [z] for z in x])

def test_check_all_is_type() -> None:
    """Check that dtypes can be checked"""
    check_all_is_type(1, 2, dtype=int)
    check_all_is_type(1, 2, 3.0, dtype=[int,float])

def test_has_assertion_error() -> None:
    """Check that error will """
    assert has_assertion_error(check_all_is_type, 1, 2, 3.4, dtype=int), 'expected True to return'
    assert has_assertion_error(check_all_pos, 1,2,3,0, strict=True), 'expected True to return'

def test_get_max_shape() -> None:
    """Check that returned shape is as expected"""
    assert get_max_shape(1, 2.0) == (1,)
    assert get_max_shape(1, np.array(2.0)) == (1,)
    assert get_max_shape(1, np.random.randn(10)) == (10,)
    assert get_max_shape(1, np.random.randn(10), np.random.randn(3,2)) == (3,2)

def test_broastcast_max_shape() -> None:
    a = 1
    b = [2,3]
    c = np.random.randn(10,2)
    d = np.random.randn(1,2)
    e = pd.Series(range(10))
    assert all([a.shape == (2,) for a in broastcast_max_shape(*[a, b])])
    assert all([a.shape == (10,2) for a in broastcast_max_shape(*[c, d])])
    assert all([a.shape == (10,2) for a in broastcast_max_shape(*[a, c])])
    assert all([a.shape == (1,2) for a in broastcast_max_shape(*[a, d])])
    assert has_assertion_error(broastcast_max_shape, *[b,c])
    assert has_assertion_error(broastcast_max_shape, *[b,e])
    assert has_assertion_error(broastcast_max_shape, *[c,e])
    assert has_assertion_error(broastcast_max_shape, *[b,d])


if __name__ == "__main__":
    # Check all functions
    test_try2list()
    test_check_all_is_type()
    test_has_assertion_error()
    test_get_max_shape()
    test_broastcast_max_shape()

    print('~~~ The test_utils.py script worked successfully ~~~')