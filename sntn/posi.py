"""
Post selection inference classes (wrapper for workhorse classes)
"""

# External
import numpy as np
# Internal
from sntn._posi import _posi_marginal_screen


"""See _posi_marginal_screen and _data_carve from sntn._posi"""
class marginal_screen(_posi_marginal_screen):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(k, y, x, **kwargs)




