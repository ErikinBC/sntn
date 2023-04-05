"""
Checks that package has configured properly, `python3 -m sntn`, will run this script. Runs the scripts contained in the readme.md.
"""

# Load dependencies
import numpy as np
import pandas as pd
from scipy.stats import norm
# from sntn.dists import SNTN, tnorm

def fun_main() -> None:
    # Ground truth
    nsim = 1000
    mu1_seq = np.linspace(50, 60, 11)
    mu2 = 70
    sig1 = 3
    sig2 = 5
    # dist1 = tnorm()
    dist2 = norm(loc=mu2, scale=sig2)

    for mu1 in mu1_seq:
        mu1

if __name__ == '__main__':
    fun_main()
    print('~~~ The sntn package was successfully compiled ~~~')
