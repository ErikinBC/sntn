"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
# Internal modules
from sntn._tnorm import _tnorm
from sntn._sntn import _SNTN

"""Inherit the entire class"""
class tnorm(_tnorm):
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int) -> None:
        super().__init__(mu, sigma2, a, b)


"""Inherit the entire class"""
class SNTN(_SNTN):
    def __init__(self) -> None:
        super().__init__()