"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
# Internal modules
from sntn._bvn import _bvn
from sntn._nts import _nts
from sntn._tnorm import _tnorm


"""tnorm (truncated normal) inherits the _tnorm class"""
class tnorm(_tnorm):
    def __init__(self, mu:float | np.ndarray | int, sigma2:float | np.ndarray | int, a:float | np.ndarray | int, b:float | np.ndarray | int) -> None:
        super().__init__(mu, sigma2, a, b)


"""Normal & Truncated Sum (NTS) inherits the _nts class"""
class nts(_nts):
    def __init__(self,  mu1:float | np.ndarray, tau21:float | np.ndarray, mu2:float | np.ndarray, tau22:float | np.ndarray, a:float | np.ndarray, b:float | np.ndarray, c1:float | np.ndarray=1, c2:float | np.ndarray=1, **kwargs) -> None:
        super().__init__(mu1, tau21, mu2, tau22, a, b, c1, c2, **kwargs)

"""Bivariate normal (BVN) inherits the _bvn class"""
class bvn(_bvn):
    def __init__(self, mu1:float | np.ndarray, sigma21:float | np.ndarray, mu2:float | np.ndarray, sigma22:float | np.ndarray, rho:float | np.ndarray, **kwargs) -> None:
        super().__init__(mu1, sigma21, mu2, sigma22, rho, **kwargs)
