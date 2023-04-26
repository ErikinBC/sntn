"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
# Internal modules
from sntn._tnorm import _tnorm
from sntn._sntn import _nts

"""Inherit the entire class"""
class tnorm(_tnorm):
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int) -> None:
        super().__init__(mu, sigma2, a, b)


"""Inherit the entire class"""
class nts(_nts):
    def __init__(self,  mu1:float or np.ndarray, tau1:float or np.ndarray, mu2:float or np.ndarray, tau2:float or np.ndarray, a:float or np.ndarray, b:float or np.ndarray, c1:float or np.ndarray=1, c2:float or np.ndarray=1) -> None:
        super().__init__(mu1, tau1, mu2, tau2, a, b, c1, c2)