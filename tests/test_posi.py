"""
Make sure PoSI classes work as expected

python3 -m tests.test_posi
python3 -m pytest tests/test_posi.py -s
"""

# External
import pytest
from sklearn.linear_model import ElasticNetCV
# Internal
from sntn.utilities.linear import dgp_sparse_yX, ols