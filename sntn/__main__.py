"""
Checks that package has configured properly, `python3 -m sntn`, will run this script. Runs the scripts contained in the readme.md.

For package updates:
# Clean up old wheels
1) rm -r dist/
2) python setup.py bdist_wheel --universal
# On some test conda env
1) twine upload --repository-url https://test.pypi.org/legacy/ dist/sntn*
2) pip uninstall sntn
3) pip install -i https://test.pypi.org/simple/ sntn==X.X.X
# Upload to PYPI https://pypi.org/project/sntn/
1) twine upload dist/sntn*
2) pip uninstall sntn
3) pip install sntn

"""

# Load dependencies
import numpy as np
from sntn.dists import nts

def fun_main() -> None:
    # Check that classic 1964 technometrics query works
    mu1, tau21 = 100, 6**2
    mu2, tau22 = 50, 3**2
    a, b = 44, np.inf
    w = 138
    dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
    expected_1964 = 0.03276
    cdf_1964 = dist_1964.cdf(w)[0]
    assert np.round(cdf_1964,5) == expected_1964, F'Expected CDF to be: {expected_1964} not {cdf_1964}, nts package did not compile properly!!'

if __name__ == '__main__':
    fun_main()
    print('~~~ The sntn package was successfully compiled ~~~')
