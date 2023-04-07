"""
Contains main distributions (i.e. SNTN)
"""

# External modules
import numpy as np
from scipy.stats import truncnorm
# Internal modules
from sntn.utilities.utils import rvec, cvec, vprint, broastcast_max_shape


class tnorm():
    def __init__(self, mu:float or np.ndarray or int, sigma2:float or np.ndarray or int, a:float or np.ndarray or int, b:float or np.ndarray or int):
        """
        Main model class for the truncated normal distribution

        Parameters
        ----------
        mu:                 Means of the unconditional normal (can be array)
        sigma2:             Variance of the truncated normal (can be array)
        a:                  Lower bounds of truncated normal (can be -infty)
        b:                  Upperbounds of truncated normal (can be +infty)

        Attributes
        ----------
        ....

        Methods
        -------
        ...

        """
        # di2 = {k: len(v) if isinstance(v,np.ndarray) | isinstance(v,list) 
        #         else 1 for k, v in di.items()}
        # self.p = max(list(di2.values()))
        # for k in di:
        #     if di2[k] == 1:
        #         di[k] = np.repeat(di[k], self.p)
        #     else:
        #         di[k] = np.array(di[k])
        # self.sig2, self.a, self.b = di['sig2'], di['a'], di['b']
        # sig = np.sqrt(di['sig2'])
        # alpha, beta = (di['a']-di['mu'])/sig, (di['b']-di['mu'])/sig
        self.dist = truncnorm()#(loc=di['mu'], scale=sig, a=alpha, b=beta)

    def cdf(self, x):
        """Wrapper for scipy.stats.truncnorm(...).cdf()"""
        return self.dist.cdf(x)

    def ppf(self, x):
        """Wrapper for scipy.stats.truncnorm(...).ppf()"""
        return self.dist.ppf(x)

    def pdf(self, x):
        """Wrapper for scipy.stats.truncnorm(...).pdf()"""
        return self.dist.pdf(x)

    def rvs(self, n, seed=None):
        """Wrapper for scipy.stats.truncnorm(...).rvs()"""
        return self.dist.rvs(n, random_state=seed)


    def get_CI(self, x, gamma:float=0.05, lb:int or float=-1000, ub:int or float=10, nline:int=25, tol:float=1e-2, imax:int=10, verbose:bool=False):
        """
        Explain CI approach
        """
        x = rvec(x)
        k = max(self.p, x.shape[1])
        # Initialize
        mu_seq = np.round(np.sinh(np.linspace(np.repeat(np.arcsinh(lb),k), 
            np.repeat(np.arcsinh(ub),k),nline)),5)
        q_seq = np.zeros(mu_seq.shape)
        q_err = q_seq.copy()
        cidx = list(range(k))
        iactive = cidx.copy()
        pidx = cidx.copy()
        j, aerr = 0, 1
        while (j<=imax) & (len(iactive)>0):
            j += 1
            vprint('------- %i -------' % j, verbose)
            # Calculate quantile range
            mus = mu_seq[:,iactive]
            if len(iactive) == 1:
                mus = mus.flatten()
            if self.p == 1:
                pidx = np.repeat(0, len(iactive))
            elif len(iactive)==1:
                pidx = iactive
            else:
                pidx = iactive
            qs = tnorm(mus, self.sig2[pidx], self.a[pidx], self.b[pidx]).ppf(gamma)
            if len(qs.shape) == 1:
                qs = cvec(qs)
            q_seq[:,iactive] = qs
            tmp_err = q_seq - x
            q_err[:,iactive] = tmp_err[:,iactive]
            istar = np.argmin(q_err**2,0)
            q_star = q_err[istar, cidx]
            mu_star = mu_seq[istar,cidx]
            idx_edge = (mu_star == lb) | (mu_star == ub)
            aerr = 100*np.abs(q_star)
            if len(aerr[~idx_edge]) > 0:
                vprint('Largest error: %0.4f' % max(aerr[~idx_edge]),verbose)
            idx_tol = aerr < tol
            iactive = np.where(~(idx_tol | idx_edge))[0]
            # Get new range
            if len(iactive) > 0:
                new_mu = np.linspace(mu_seq[np.maximum(0,istar-1),cidx],
                                    mu_seq[np.minimum(istar+1,nline-1),cidx],nline)
                mu_seq[:,iactive] = new_mu[:,iactive]
        mu_star = mu_seq[istar, cidx]
        # tnorm(mu=mu_star, sig2=self.sig2, a=self.a, b=self.b).ppf(gamma)
        return mu_star


