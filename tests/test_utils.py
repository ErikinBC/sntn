"""Test utilities functions: python3 -m tests.test_utils"""

import numpy as np
from sntn.utilities.utils import check_all_is_type, try2list, has_assertion_error

def test_try2list() -> None:
    """ Check that try2list returns lists as expected """
    x = [1,2,3]
    assert try2list(x) == x
    assert try2list(np.array(x)) == x
    assert all([try2list(z) == [z] for z in x])


def test_check_all_is_type() -> None:
    check_all_is_type(1, 2, dtype=int)
    check_all_is_type(1, 2, 3.0, dtype=[int,float])


if __name__ == "__main__":
    # Check all functions
    test_try2list()
    test_check_all_is_type()

    print('~~~ The test_utils.py script worked successfully ~~~')