"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
from copy import deepcopy
from scipy.stats import truncnorm, norm
from scipy.optimize import root, minimize_scalar, minimize, root_scalar
# Internal modules
from sntn.utilities.utils import broastcast_max_shape, grad_clip_abs
from sntn.utilities.grad import _log_gauss_approx, _log_diff


