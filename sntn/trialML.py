"""
Classes to support example applications
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from scipy.stats import t as stud_t
# Internal modules
from sntn.dists import nts


class two_stage():
    def __init__(self, n:int, m:int, gamma:float, alpha:float=0.05, pool:bool=True, student:bool=True) -> None:
        """
        Wrapper for performing a two-stage regression problem
        """
        # Input checks and assign
        assert (n > 1) and (m >= 1) and (gamma > 0) & (gamma < 1)
        self.n, self.m, self.gamma = n, m, gamma
        self.alpha, self.pool = alpha, pool

        # Calculate degres of freedom
        self.dof_S, self.dof_T = n - 1, m - 1
        if self.pool:
            self.dof_T = n + m - 1
        if student:
            self.phi_inv = stud_t(df=self.dof_S).ppf(1-gamma)
        else:
            self.phi_inv = norm.ppf(1-gamma)
        mn_ratio = m / n
        mu_2stage = np.array([0, -np.sqrt(mn_ratio)*self.phi_inv])
        tau_2stage = np.sqrt([1, mn_ratio])
        self.H0 = nts(mu=mu_2stage,tau=tau_2stage, a=0, b=np.infty)
        self.HA = nts(mu=mu_2stage,tau=tau_2stage, a=-np.infty, b=0)
        self.t_alpha = self.H0.ppf(alpha)
        self.power = self.HA.cdf(self.t_alpha)

    def rvs(self, nsim, delta, sigma2, seed=None):
        if seed is None:
            seed = nsim
        np.random.seed(seed)
        delta1 = delta + np.sqrt(sigma2/self.n)*np.random.randn(nsim)
        delta2 = delta + np.sqrt(sigma2/self.m)*np.random.randn(nsim)
        sigS = np.sqrt(sigma2*chi2(df=self.dof_S).rvs(nsim)/self.dof_S)
        sigT = np.sqrt(sigma2*chi2(df=self.dof_T).rvs(nsim)/self.dof_T)
        delta0 = delta1 + (sigS/np.sqrt(self.n))*self.phi_inv
        shat = (delta2 - delta0)/(sigT/np.sqrt(self.m))
        df = pd.DataFrame({'shat':shat, 'd0hat':delta0})
        return df
