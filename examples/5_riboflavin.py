"""
Show CIs for riboflavin dataset
"""

# External
import os
import numpy as np
import pandas as pd
# Internal
from parameters import dir_figures
from sntn.posi import lasso
from sntn.utilities.utils import get_CI, pn_labeller

