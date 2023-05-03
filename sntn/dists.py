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
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int) -> None:
        super().__init__(mu, sigma2, a, b)


"""Normal & Truncated Sum (NTS) inherits the _nts class"""
class nts(_nts):
    def __init__(self,  mu1:float or np.ndarray, tau21:float or np.ndarray, mu2:float or np.ndarray, tau22:float or np.ndarray, a:float or np.ndarray, b:float or np.ndarray, c1:float or np.ndarray=1, c2:float or np.ndarray=1, **kwargs) -> None:
        super().__init__(mu1, tau21, mu2, tau22, a, b, c1, c2, **kwargs)

"""Bivariate normal (BVN) inherits the _bvn class"""
class bvn(_bvn):
    def __init__(self, mu1:float or np.ndarray, sigma21:float or np.ndarray, mu2:float or np.ndarray, sigma22:float or np.ndarray, rho:float or np.ndarray, **kwargs) -> None:
        super().__init__(mu1, sigma21, mu2, sigma22, rho, **kwargs)
