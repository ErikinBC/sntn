import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm, truncnorm, t, chi2
from scipy.stats import multivariate_normal as MVN
from scipy.linalg import cholesky
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from time import time


class BVN():
    def __init__(self, mu, sigma, rho):
        """
        mu: array of means
        sigma: array of variances
        rho: correlation coefficient
        """
        if isinstance(mu,list):
            mu, sigma = np.array(mu), np.array(sigma)
        assert mu.shape[0]==sigma.shape[0]==2
        assert np.abs(rho) <= 1
        self.mu = mu.reshape([1,2])
        self.sigma = sigma.flatten()
        od = rho*np.sqrt(sigma.prod())
        self.rho = rho
        self.Sigma = np.array([[sigma[0],od],[od, sigma[1]]])
        self.A = cholesky(self.Sigma) # A.T.dot(A) = Sigma

    # size=1000;seed=1234 # del size, seed
    def rvs(self, size, seed=None):
        """
        size: number of samples to simulate
        seed: to pass onto np.random.seed
        """
        np.random.seed(seed)
        X = np.random.randn(size,2)
        Z = self.A.T.dot(X.T).T + self.mu
        return Z

    def imills(self, a):
        return norm.pdf(a)/norm.cdf(-a)

    def sheppard(self, theta, h, k):
        return (1/(2*np.pi))*np.exp(-0.5*(h**2+k**2-2*h*k*np.cos(theta))/(np.sin(theta)**2))

    # h, k = -2, -np.infty
    def orthant(self, h, k, method='scipy'):
        # P(X1 >= h, X2 >=k)
        assert method in ['scipy','cox','sheppard']
        if isinstance(h,int) or isinstance(h, float):
            h, k = np.array([h]), np.array([k])
        else:
            assert isinstance(h,np.ndarray) and isinstance(k,np.ndarray)
        assert len(h) == len(k)
        # assert np.all(h >= 0) and np.all(k >= 0)
        # Calculate the number of standard deviations away it is        
        Y = (np.c_[h, k] - self.mu)/np.sqrt(self.sigma)
        Y1, Y2 = Y[:,0], Y[:,1]
        
        # (i) scipy: L(h, k)=1-(F1(h)+F2(k))+F12(h, k)
        if method == 'scipy':
            sp_bvn = MVN([0, 0],[[1,self.rho],[self.rho,1]])
            pval = 1+sp_bvn.cdf(Y)-(norm.cdf(Y1)+norm.cdf(Y2))
            return pval 

        # A Simple Approximation for Bivariate and Trivariate Normal Integrals
        if method == 'cox':
            mu_a = self.imills(Y1)
            root = np.sqrt(1-self.rho**2)
            xi = (self.rho * mu_a - Y2) / root
            pval = norm.cdf(-Y1) * norm.cdf(xi)
            return pval

        if method == 'sheppard':
            pval = np.array([quad(self.sheppard, np.arccos(self.rho), np.pi, args=(y1,y2))[0] for y1, y2 in zip(Y1,Y2)])
            return pval



# mu=np.array([[0,0],[1,1]]);tau=np.array([[1,1],[1,1]]);a=np.array([1,1]); b=np.array([np.inf,np.inf])
# mu = np.repeat(0,20).reshape([10,2]); tau=np.linspace(1,2,21)[:-1].reshape([10,2])
# a = np.repeat(1,10); b=np.repeat(np.inf,10)
# self = NTS(mu, tau, a, b)
# mu, tau, a, b = [0,0], [1,1], 1, np.inf
# self = NTS(mu, tau, a, b)
class NTS():
    def __init__(self, mu, tau, a, b):
        """
        mu: matrix/array of means
        tau: matrix/array of standard errors
        a: array of lower bounds
        b: array of upper bounds
        """
        if not (isinstance(mu,np.ndarray) & isinstance(tau,np.ndarray)):
            mu, tau = np.array(mu), np.array(tau)
        assert mu.shape == tau.shape
        if len(mu.shape) == 2:
            assert mu.shape[1] == tau.shape[1] == 2
            a, b = np.array(a), np.array(b)
        else:
            assert mu.shape[0] == tau.shape[0] == 2
            mu, tau = rvec(mu), rvec(tau)
            a, b, = np.array([a]), np.array([b])
        self.r = mu.shape[0]
        assert self.r == len(a) == len(b)
        self.mu, self.tau, self.a, self.b = mu, tau, a.flatten(), b.flatten()
        # Truncated normal (Z2)
        self.alpha = (self.a - self.mu[:,1]) / self.tau[:,1]
        self.beta = (self.b - self.mu[:,1]) / self.tau[:,1]
        self.Z = norm.cdf(self.beta) - norm.cdf(self.alpha)
        self.Q = norm.pdf(self.alpha) - norm.pdf(self.beta)
        # Average will be unweighted combination of the two distributions
        self.mu_W = self.mu[:,0] + self.mu[:,1] + self.tau[:,1]*self.Q/self.Z
        if np.prod(self.mu_W.shape) == 1:
            self.mu_W = self.mu_W[0]
        # Distributions
        self.dist_X1 = norm(loc=self.mu[:,0], scale=self.tau[:,0])
        self.dist_X2 = truncnorm(a=self.alpha, b=self.beta, loc=self.mu[:,1], scale=self.tau[:,1])
        # W
        self.theta1 = self.mu.sum(1)
        self.theta2 = self.mu[:,1]
        self.sigma1 = np.sqrt(np.sum(self.tau**2,1))
        self.sigma2 = self.tau[:,1]
        self.rho = self.sigma2/self.sigma1
        # Initialize BVN for CDF
        self.di_BVN = {i:BVN(mu=[0,0],sigma=[1,1],rho=self.rho[i]) for i in range(self.r)}

    def reshape(self, x):
        if isinstance(x, list) | isinstance(x, pd.Series):
            x = np.array(x)
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        nele = np.prod(x.shape)
        nr = int(nele / self.r)
        if nr == 0:
            assert len(x) == 1
            x = rvec(np.repeat(x,self.r))
            nr += 1
        else:
            x = x.reshape([nr, self.r])
        return x

    # x = np.c_[np.repeat(2,10),np.repeat(3,10)].T
    def pdf(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, float) or isinstance(x, int):
            x = np.array([x])
        term1 = self.sigma1 * self.Z
        m1 = (x - self.theta1) / self.sigma1
        term2 = (self.beta-self.rho*m1)/np.sqrt(1-self.rho**2)
        term3 = (self.alpha-self.rho*m1)/np.sqrt(1-self.rho**2)
        f = norm.pdf(m1)*(norm.cdf(term2) - norm.cdf(term3)) / term1
        if np.prod(f.shape) == 1:
            f = f[0]
        return f

    def cdf(self, x, method='scipy'):
        """
        THE SIZE OF x NEEDS TO MATCH VECTORIZIATION
        if self.r > 1:  x ~ [nr, self.r]
        """
        x = self.reshape(x)
        nr, nc = x.shape
        m1 = (x - self.theta1) / self.sigma1
        alpha_seq = np.where(self.alpha == -np.infty, -10, self.alpha)
        beta_seq = np.where(self.beta == np.infty, +10, self.beta)
        if self.r == 1:
            assert nc == 1
            alpha_seq = cvec(np.repeat(alpha_seq,nr))
            beta_seq = cvec(np.repeat(beta_seq,nr))
        else:  # (self.r > 1):
            alpha_seq, beta_seq = np.tile(alpha_seq, [nr,1]), np.tile(beta_seq, [nr,1])
        # Looping must be done because self.rho varies by di_BVN...
        orthant1 = np.vstack([self.di_BVN[i].orthant(m1[:,i],alpha_seq[:,i],method=method) for i in range(self.r)]).T
        orthant2 = np.vstack([self.di_BVN[i].orthant(m1[:,i],beta_seq[:,i],method=method) for i in range(self.r)]).T
        orthant = 1 - (orthant1 - orthant2)/self.Z
        if np.any(np.array(orthant.shape)==1):
            orthant = orthant.flatten()
        if np.prod(orthant.shape) == 1:
            orthant = orthant[0]
        return orthant

    # self = dist_H0_NTS; x=bhat12
    # gamma=alpha/2;verbose=True; nline=10; imax=10; kse=8; tol=0.01
    def CI(self, x, gamma, nline=25, imax=10, kse=8, tol=0.01, verbose=False):
        x = rvec(x)
        nc = x.shape[1]
        lb, ub = self.mu_W-kse*self.sigma1, self.mu_W+kse*self.sigma1
        mu_seq = np.linspace(lb,ub,nline)
        if (mu_seq.shape[1]==1) & (nc>1):
            mu_seq = np.tile(mu_seq,[1,nc])
            tau = np.tile(self.tau,[nc,1])
            a, b = np.repeat(self.a,nc), np.repeat(self.b,nc)
        else:
            tau, a, b, = self.tau, self.a, self.b
        cidx = list(range(nc))
        cidx_flat = pd.Series(np.tile(cidx,[nline,1]).flatten())
        p_err_flat = np.ones(cidx_flat.shape)
        iactive = cidx.copy()
        # tau, a, b don't change
        taus = np.tile(tau,[nline,1])
        aas = np.tile(a,[nline,1]).flatten()
        bbs = np.tile(b,[nline,1]).flatten()
        xs = np.tile(x,[nline,1])
        xs_flat = xs.flatten()
        # Loop it
        j = 0
        while (j<=imax) & (len(iactive)>0):
            j += 1
            vprint('Iteration %i (%i active)' % (j, len(iactive)),verbose)
            mus_flat = mu_seq.flatten()
            mus_mat = np.c_[mus_flat,mus_flat]
            iactive_flat = cidx_flat.isin(iactive)
            # pd.DataFrame({'tau':taus[:,0],'mu':mus_flat}).groupby('tau').mu.describe()[['min','max']]
            # np.c_[mu_seq.min(0),mu_seq.max(0)]
            dist_j = NTS(mus_mat[iactive_flat],taus[iactive_flat],aas[iactive_flat],bbs[iactive_flat])
            p_err_flat[iactive_flat] = dist_j.cdf(xs_flat[iactive_flat]) - gamma
            p_err = p_err_flat.reshape(mu_seq.shape)
            # Make sure there is an intermediate value
            istar = np.nanargmin(p_err**2,0)
            assert np.all((istar != 0) & (istar != (nline-1)))
            a_p_err = 100*np.abs(p_err[istar,cidx])
            iactive = np.where(~(a_p_err < tol))[0]
            if len(iactive) > 0:
                mus = mus_flat.reshape(mu_seq.shape)
                new_mus = np.linspace(mus[istar-1,cidx],mus[istar+1,cidx],nline)
                mu_seq[:,iactive] = new_mus[:,iactive]
        # Get the final value
        mu_star = mu_seq[istar, cidx]
        return mu_star

    def ppf(self, p):
        p = self.reshape(p)
        def mfun(w, pp):
            pval = self.cdf(w.reshape(pp.shape)).reshape(pp.shape)
            err2 = np.sum((pval - pp)**2)
            return err2
        # Initialize guess with conservative standard normal
        x0 = self.mu_W+norm.ppf(p)*self.tau[:,0]        
        # x0 = np.zeros(p.shape) + self.mu_W
        res = minimize(fun=mfun,x0=x0,args=(p),method='L-BFGS-B')
        w = res.x.reshape(p.shape)
        if np.any(np.array(w.shape)==1):
            w = w.flatten()
        if np.prod(w.shape) == 1:
            w = w[0]
        return w

    def rvs(self, n, seed=1234):
        r1 = self.dist_X1.rvs(size=n,random_state=seed)
        r2 = self.dist_X2.rvs(size=n,random_state=seed)
        return r1 + r2

class tnorm():
    def __init__(self, mu, sig2, a, b):
        di = {'mu':mu, 'sig2':sig2, 'a':a, 'b':b}
        di2 = {k: len(v) if isinstance(v,np.ndarray) | isinstance(v,list) 
                else 1 for k, v in di.items()}
        self.p = max(list(di2.values()))
        for k in di:
            if di2[k] == 1:
                di[k] = np.repeat(di[k], self.p)
            else:
                di[k] = np.array(di[k])
        self.sig2, self.a, self.b = di['sig2'], di['a'], di['b']
        sig = np.sqrt(di['sig2'])
        alpha, beta = (di['a']-di['mu'])/sig, (di['b']-di['mu'])/sig
        self.dist = truncnorm(loc=di['mu'], scale=sig, a=alpha, b=beta)

    def cdf(self, x):
        return self.dist.cdf(x)

    def ppf(self, x):
        return self.dist.ppf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def rvs(self, n, seed=None):
        return self.dist.rvs(n, random_state=seed)

    def CI(self, x, gamma=0.05, lb=-1000, ub=10, nline=25, tol=1e-2, imax=10, verbose=False):
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



