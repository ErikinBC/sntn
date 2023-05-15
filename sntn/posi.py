"""
Post selection inference classes (wrapper for workhorse classes)
"""

# External
import numpy as np
# Internal
from sntn._screening import _posi_marginal_screen
from sntn._lasso import _posi_lasso


"""See _posi_lasso and _data_carve from sntn._posi"""
class lasso(_posi_lasso):
    def __init__(self, lam:float, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(lam, y, x, **kwargs)



"""See _posi_marginal_screen and _data_carve from sntn._posi"""
class marginal_screen(_posi_marginal_screen):
    def __init__(self, k:int, y:np.ndarray, x:np.ndarray, **kwargs):
        super().__init__(k, y, x, **kwargs)




