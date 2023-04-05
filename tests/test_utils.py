"""Test utilities functions: python3 -m tests.test_utils"""

import numpy as np
from sntn.utilities.utils import check_all_is_type, try2list, has_assertion_error, check_all_pos

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
    

if __name__ == "__main__":
    # Check all functions
    test_try2list()
    test_check_all_is_type()
    test_has_assertion_error()

    print('~~~ The test_utils.py script worked successfully ~~~')