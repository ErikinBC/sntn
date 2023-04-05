import os
import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm, chi2, t
from plotnine import *
from timeit import timeit
from time import time
from classes import cvec

from statsmodels.stats.proportion import proportion_confint as propCI

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from classes import BVN, NTS, two_stage, ols, dgp_yX, tnorm, rvec

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base,'figures')

#######################################
# --- (2) BVN INTEGRATION METHODS --- #

# Set up a BVN
mu = np.array([1,2])
sigma = np.array([0.5,2])
dist_BVN = BVN(mu,sigma,rho=0.4)
nrun = 10000

# Check that different approaches line up
h_seq = np.linspace(mu[0],mu[0]+2*sigma[0],5)
k_seq = np.linspace(mu[1],mu[1]+2*sigma[1],5)
n_std = np.arange(len(h_seq))/2
methods = ['scipy','cox','sheppard']

holder = []
for h in h_seq:
    for k in k_seq:
        print('h: %0.1f, k: %0.1f' % (h,k))
        ptime = np.zeros([len(methods),2])
        for i, method in enumerate(methods):
            tmp_pv = dist_BVN.orthant(h, k, method)
            tmp_time = timeit('dist_BVN.orthant(h, k, method)',number=nrun,globals=globals())
            ptime[i] = [tmp_pv, tmp_time]
        tmp_df = pd.DataFrame(ptime,columns=['pval','rtime']).assign(h=h,k=k,method=methods)
        holder.append(tmp_df)
df_pval = pd.concat(holder).reset_index(None,True)
df_pval = df_pval.assign(h=lambda x: (x.h-mu[0])/sigma[0], k=lambda x: (x.k-mu[1])/sigma[1])
df_pval = df_pval.assign(xlbl=lambda x: '('+x.h.astype(str)+','+x.k.astype(str)+')')

# Repeat timing for vectorized operations
nrun = 1000
ll = range(50,501,50)
holder = []
methods2 = ['cox','scipy']
for l in ll:
    print(l)
    h = np.arange(l) / l
    k = h + 1
    tmp = np.repeat(0.0,len(methods2))
    for i,method in enumerate(methods2):
        tmp[i] = timeit('dist_BVN.orthant(h, k, method)',number=nrun,globals=globals())
    holder.append(pd.DataFrame({'method':methods2,'rtime':tmp,'ll':l}))
df_vec = pd.concat(holder).reset_index()

posd = position_dodge(0.5)
# Vectorized run times
gg_rvec = (ggplot(df_vec,aes(x='ll',y='rtime',color='method')) + 
    theme_bw() + geom_line() + geom_point() + 
    labs(x='Vector length',y='Run time for 1K calls') + 
    scale_color_manual(name='CDF method',values=["#F8766D","#00BA38"]) + 
    scale_x_continuous(breaks=list(np.array(ll))) + 
    guides(color=False))
gg_rvec.save(os.path.join(dir_figures,'gg_rvec.png'),width=4,height=4.5)

# P-values
gg_pval = (ggplot(df_pval,aes(x='xlbl',y='-np.log2(pval)',color='method')) + 
    theme_bw() + scale_color_discrete(name='CDF method') + 
    geom_point(size=1,position=posd) + 
    labs(x='(h,k) std dev',y='-log2 orthant probability') + 
    theme(axis_text_x=element_text(angle=90)))
gg_pval.save(os.path.join(dir_figures,'gg_pval.png'),width=7,height=4.5)

# Run times
gg_rtime = (ggplot(df_pval,aes(x='xlbl',y='rtime',color='method')) + 
    theme_bw() + scale_color_discrete(name='CDF method') +  
    geom_point(size=1,position=posd) + 
    labs(x='(h,k) std dev',y='Run time for 10K calls') + 
    theme(axis_text_x=element_text(angle=90)))
gg_rtime.save(os.path.join(dir_figures,'gg_rtime.png'),width=7,height=4.5)

####################################
# --- (3) NORMAL-TRUNCATED-SUM --- #

# --- NTS --- #

# Check vectorization works for cdf, pdf, and ppf
# r == 1 & x == 1
nts1 = NTS(mu=[0,0],tau=[1,1],a=1,b=np.inf)
nts1.pdf(x=2); nts1.cdf(x=2); nts1.ppf(p=0.025)
nts1.CI(x=nts1.ppf(p=[alpha/2,1-alpha/2]),gamma=0.975)
nts1.CI(x=nts1.ppf(p=0.025),gamma=1-0.975)
# r == 1 & x > 1
nts1.pdf(x=[2,2]); nts1.cdf(x=[2,2]); nts1.ppf(p=[0.2,0.5])
# r > 1 & x == 1
nts2 = NTS(mu=[[0,0],[1,1]],tau=[[1,1],[1,1]],a=[1,1],b=[np.inf,np.inf])
nts2.pdf(x=2); nts2.cdf(x=2); nts2.ppf(p=0.5)
# r > 1 & x == r
nts2.pdf(x=[2,3]); nts2.cdf(x=[1.5,2.8]); nts2.ppf(p=[0.5,0.5])
# r > 1 & x == c*r
xx = np.arange(6).reshape([3,2])
pp = np.tile([0.25,0.5,0.75],[2,1]).T
nts2.pdf(xx); nts2.cdf(xx); nts2.ppf(pp)

tmp_mu = np.array([[0,0],[1,1],[2,2]])
tmp_tau = tmp_mu + 0.5
nts3 = NTS(tmp_mu, tmp_tau, np.repeat(1,3),np.repeat(np.inf,3))
nts3.pdf(x=2); nts3.cdf(x=2); nts3.ppf(p=0.025); nts3.ppf(p=0.975)
x = nts3.ppf(p=0.025)
nts3.CI(x,1-alpha/2)

# # Sample distribution
# nsim = 1000
# nts_rvs = NTS(mu=[0,0],tau=[1,1],a=1,b=np.inf)
# gauss_rvs = norm(loc=nts_rvs.theta1,scale=nts_rvs.sigma1)
# d1 = pd.DataFrame({'x':nts_rvs.rvs(nsim,seed=nsim),'tt':'NTS'})
# d2 = pd.DataFrame({'x':gauss_rvs.rvs(nsim,random_state=nsim),'tt':'Gaussian'})
# df = pd.concat([d1,d2]).reset_index(None,True)
# gg = (ggplot(df,aes(x='x',fill='tt')) + theme_bw() + 
#     geom_histogram(alpha=0.5,color='black',bins=25,position='identity') )
# gg.save(os.path.join(dir_figures,'gg_tmp.png'),height=4,width=5)

# Demonstrated with example
mu1, tau1 = 1, 1
mu2, tau2, a, b = 1, 2, -1, 4
mu, tau = np.array([mu1, mu2]), np.array([tau1,tau2])
dist_NTS = NTS(mu=mu,tau=tau, a=a, b=b)
n_iter = 100000
W_sim = dist_NTS.rvs(n=n_iter,seed=1)
mu_sim, mu_theory = W_sim.mean(),dist_NTS.mu_W
xx = np.linspace(-5*mu.sum(),5*mu.sum(),n_iter)
mu_quad = np.sum(xx*dist_NTS.pdf(xx)*(xx[1]-xx[0]))
methods = ['Empirical','Theory', 'Quadrature']
mus = [mu_sim, mu_theory, mu_quad]
# Compare
print(pd.DataFrame({'method':methods,'mu':mus}))

# Do quadrature approximation
dat_W_sim = pd.DataFrame({'W':W_sim,'f':dist_NTS.pdf(W_sim)})

# Plot empirical frequency to theory
gg_Wsim = (ggplot(dat_W_sim) + theme_bw() +
    labs(x='W',y='Simulation density') + 
    ggtitle('Blue curve shows Eq. (3)') + 
    stat_bin(aes(x='W',y=after_stat('density')),bins=40,geom='geom_histogram',
        fill='red',alpha=0.5,color='black') + 
    stat_function(aes(x='W'),fun=dist_NTS.pdf,color='blue'))
gg_Wsim.save(os.path.join(dir_figures,'gg_Wsim.png'),width=5,height=4)

# Create a P-P plot
p_seq = np.arange(0.005,1,0.005)
q_seq = np.quantile(W_sim,p_seq)
p_theory = dist_NTS.cdf(q_seq)
q_theory = dist_NTS.ppf(p_seq)

tmp1 = pd.DataFrame({'tt':'Percentile','theory':p_theory,'emp':p_seq}).reset_index()
tmp2 = pd.DataFrame({'tt':'Quantile','theory':q_theory,'emp':q_seq}).reset_index()
dat_ppqq = pd.concat([tmp1,tmp2]).reset_index(None, True)

# Plot empirical frequency to theory
gg_ppqq = (ggplot(dat_ppqq,aes(x='theory',y='emp')) + 
    theme_bw() + geom_point(size=0.5) + 
    geom_abline(slope=1,intercept=0,linetype='--',color='blue') + 
    facet_wrap('~tt',scales='free') + 
    labs(x='Theory',y='Empirical') + 
    theme(subplots_adjust={'wspace': 0.15}))
gg_ppqq.save(os.path.join(dir_figures,'gg_ppqq.png'),width=8,height=3.5)


#####################################
# --- (4B) TWO-STAGE REGRESSION --- #

delta, sigma2 = 2, 4
n, m = 100, 100
gamma, alpha = 0.01, 0.05
nsim = 10000000
p_seq = np.round(np.arange(0.01,1,0.01),2)
two_stage(n=n, m=m, gamma=gamma, alpha=alpha, pool=True).power

# --- (A) CALCULATE N=M=100 PP-QQ PLOT --- #
dist_2s = two_stage(n=n, m=m, gamma=gamma, alpha=alpha, pool=True)
print('Power: %0.3f' % dist_2s.power)
df_2s = dist_2s.rvs(nsim=nsim, delta=delta, sigma2=sigma2)
df_2s = df_2s.assign(Null=lambda x: x.d0hat < delta)
df_2s = df_2s.assign(reject=lambda x: x.shat < dist_2s.t_alpha)
df_2s.groupby('Null').reject.mean()

qq_emp = df_2s.groupby('Null').apply(lambda x: pd.DataFrame({'pp':p_seq,'qq':np.quantile(x.shat,p_seq)}))
qq_emp = qq_emp.reset_index().drop(columns='level_1')
qq_theory_H0 = dist_2s.H0.ppf(p_seq)
qq_theory_HA = dist_2s.HA.ppf(p_seq)
tmp1 = pd.DataFrame({'pp':p_seq,'theory':qq_theory_H0,'Null':True})
tmp2 = pd.DataFrame({'pp':p_seq,'theory':qq_theory_HA,'Null':False})
qq_pp = qq_emp.merge(pd.concat([tmp1, tmp2]))
qq_pp = qq_pp.melt(['pp','Null'],['qq','theory'],'tt')

gtit = 'n=%i, m=%i, nsim=%i' % (n, m, nsim)
gg_qp_2s = (ggplot(qq_pp,aes(x='pp',y='value',color='tt')) + 
    theme_bw() + geom_line() +  ggtitle(gtit) + 
    facet_wrap('~Null',scales='free_y',labeller=label_both,ncol=1) + 
    labs(x='Percentile',y='Quantile') + 
    theme(legend_position=(0.3,0.8)) + 
    scale_color_discrete(name='Method',labels=['Empirical','Theory']))
gg_qp_2s.save(os.path.join(dir_figures,'gg_qp_2s.png'),width=4,height=7)

# --- (B) POWER AS GAMMA VARIES --- #
gamma_seq = np.round(np.arange(0.01,0.21,0.01),2)
power_theory = np.array([two_stage(n=n, m=m, gamma=g, alpha=alpha, pool=False).power for g in gamma_seq])
ub_theory = delta + np.sqrt(sigma2/n)*t(df=n-1).ppf(1-gamma_seq)
power_emp, ub_emp = np.zeros(power_theory.shape), np.zeros(ub_theory.shape)
for i, g in enumerate(gamma_seq):
    print('%i of %i' % (i+1, len(gamma_seq)))
    tmp_dist = two_stage(n=n, m=m, gamma=g, alpha=alpha, pool=False)
    tmp_sim = tmp_dist.rvs(nsim=nsim, delta=delta, sigma2=sigma2)
    tmp_sim = tmp_sim.assign(Null=lambda x: x.d0hat < delta, 
                reject=lambda x: x.shat < tmp_dist.t_alpha)
    power_emp[i] = tmp_sim.query('Null==False').reject.mean()
    ub_emp[i] = tmp_sim.d0hat.mean()

tmp1 = pd.DataFrame({'tt':'theory','gamma':gamma_seq,'power':power_theory,'ub':ub_theory})
tmp2 = pd.DataFrame({'tt':'emp','gamma':gamma_seq,'power':power_emp,'ub':ub_emp})
dat_gamma = pd.concat([tmp1, tmp2]).melt(['tt','gamma'],None,'msr')
di_msr = {'power':'Power','ub':'delta0'}

gg_gamma = (ggplot(dat_gamma,aes(x='gamma',y='value',color='tt')) + 
    theme_bw() + geom_line() +  ggtitle(gtit) + 
    facet_wrap('~msr',scales='free_y',ncol=1,labeller=labeller(msr=di_msr)) + 
    labs(x='gamma',y='Value') + 
    theme(legend_position=(0.6,0.8)) + 
    scale_x_continuous(limits=[0,0.2]) + 
    scale_color_discrete(name='Method',labels=['Empirical','Theory']))
gg_gamma.save(os.path.join(dir_figures,'gg_gamma.png'),width=4,height=7)

# --- (C) POWER AS N = K - M VARIES --- #

k = n + m
n_seq = np.arange(5,k,5)
dat_nm = pd.concat([pd.DataFrame({'n':nn,'m':k-nn,
    'power':two_stage(n=nn, m=k-nn, gamma=gamma, alpha=alpha, pool=True).power},index=[nn]) for nn in n_seq])
dat_nm = dat_nm.reset_index(None,True).assign(ub=delta + np.sqrt(sigma2/n_seq)*t(df=n_seq-1).ppf(1-gamma))
dat_nm = dat_nm.melt(['n','m'],None,'msr')

gg_nm = (ggplot(dat_nm,aes(x='n',y='value')) + 
    theme_bw() + geom_line() +  
    ggtitle('gamma=0.01, m=200-n') + 
    facet_wrap('~msr',scales='free_y',ncol=1,labeller=labeller(msr=di_msr)) + 
    labs(x='n',y='Value'))
gg_nm.save(os.path.join(dir_figures,'gg_nm.png'),width=4,height=7)


#############################
# --- (4A) QUERIES 1964 --- #
# X~N(100,6), Y~TN(50,3,44,Inf)

mu1, tau1 = 100, 6
mu2, tau2, a, b = 50, 3, 44, np.inf
mu, tau = np.array([mu1, mu2]), np.array([tau1,tau2])
dist_A = NTS(mu=mu,tau=tau, a=a, b=b)
dist_A.cdf(138)


#############################
# --- (4C) DATA CARVING --- #

mu1, mu2 = 0, 0
v1, v2 = 1.5, 2.5
a, b = 1, np.inf
nsim = 100000
alpha = 0.05
p_alpha = [alpha/2, 1-alpha/2]
# CI check for NTS
Z1 = norm(loc=mu1,scale=v1).rvs(nsim,random_state=nsim)
Z2 = tnorm(mu2,v2**2,a,b).rvs(nsim,seed=nsim)
W = Z1 + Z2
dist_NTS = NTS(mu=[mu1,mu2],tau=[v1,v2],a=a, b=b)
q_NTS = np.array([dist_NTS.ppf(p) for p in p_alpha])
print(np.round(pd.DataFrame({'q_theory':q_NTS,'q_emp':np.quantile(W,p_alpha)}),2))
CI_NTS = pd.DataFrame({'q':q_NTS,'lb':dist_NTS.CI(q_NTS,1-alpha/2),'ub':dist_NTS.CI(q_NTS,alpha/2).flat})
print(np.round(CI_NTS,2))

# CI check for TN
dist_TN = tnorm(mu=0, sig2=1, a=1, b=np.inf)
q_TN = dist_TN.ppf([alpha/2, 1-alpha/2])
CI_TN = pd.DataFrame({'q':q_TN,'lb':dist_TN.CI(q_TN,1-alpha/2),'ub':dist_TN.CI(q_TN,alpha/2)})
np.round(CI_TN,2)

alpha = 0.05
cutoff = 0.1
n, p, sig2 = 100, 20, 1
m = int(n/2)
beta_null = np.repeat(0, p)
nsim = 1000
holder_SI, holder_ols, holder_carv = [], [], []
np.random.seed(nsim)
for i in range(nsim):
    if (i+1) % 100 == 0:
        print(i+1)
    resp, xx = dgp_yX(n=n, p=p, snr=1, b0=0, seed=i)
    # (i) Data splitting
    mdl1 = ols(y=resp[:m], X=xx[:m], has_int=False, sig2=sig2)
    abhat1 = np.abs(mdl1.bhat)
    M1 = abhat1 > cutoff
    if M1.sum() > 0:
        mdl2 = ols(y=resp[m:], X=xx[m:,M1], has_int=False, sig2=sig2)
        tmp_ols = pd.DataFrame({'sim':i,'bhat':mdl2.bhat,'Vjj':np.diagonal(mdl2.covar)})
        holder_ols.append(tmp_ols)
    # (iii) Data carving
    if M1.sum() > 0:
        tmp_carv = pd.DataFrame({'sim':i,'bhat1':mdl1.bhat[M1],'bhat2':mdl2.bhat,
                      'Vm':np.diagonal(mdl1.covar)[M1],'Vk':np.diagonal(mdl2.covar)})
        holder_carv.append(tmp_carv)    
    # (ii) Selective inference
    mdl = ols(y=resp, X=xx, has_int=False, sig2=sig2)
    M = np.abs(mdl.bhat)>cutoff
    if M.sum() > 0:        
        tmp_M = pd.DataFrame({'sim':i,'bhat':mdl.bhat[M],'Vjj':np.diagonal(mdl.covar)[M]})
    holder_SI.append(tmp_M)
# In NTS?
dat_NTS = pd.concat(holder_carv).reset_index(None,True).assign(tt='NTS')
# dat_NTS = dat_NTS.assign(pval=np.NaN,lb=np.NaN,ub=np.NaN)
# Assume we're in positive region, and then swap for negative signs
a_seq = np.repeat(cutoff, len(dat_NTS))
b_seq = np.repeat(np.inf, len(dat_NTS))
tau_seq = np.sqrt(np.c_[dat_NTS.Vk, dat_NTS.Vm])
mu_seq = np.zeros(tau_seq.shape)
dist_H0_NTS = NTS(mu_seq, tau_seq, a_seq, b_seq)
bhat12 = (dat_NTS.bhat1.abs() + dat_NTS.bhat2).values
pval_NTS = dist_H0_NTS.cdf(bhat12)
pval_NTS = 2*np.minimum(pval_NTS,1-pval_NTS)
assert np.abs(alpha - np.mean(pval_NTS < alpha)) < 2*np.sqrt(alpha*(1-alpha) / len(dat_NTS))
ub_NTS = dist_H0_NTS.CI(bhat12,gamma=alpha/2,nline=10,verbose=True)
lb_NTS = dist_H0_NTS.CI(bhat12,gamma=1-alpha/2,nline=10,verbose=True)
tmp_plu = pd.DataFrame(np.c_[np.sign(dat_NTS.bhat1),pval_NTS, lb_NTS, ub_NTS],columns=['sbhat','pval','lb','ub'])
tmp_plu.loc[tmp_plu.sbhat==-1,['lb','ub']] = (-1*tmp_plu.loc[tmp_plu.sbhat==-1,['lb','ub']]).iloc[:,[1,0]].values
tmp_plu = tmp_plu.assign(reject=lambda x: x.pval<alpha, cover=lambda x: np.sign(x.lb)!=np.sign(x.ub))
dat_NTS_inf = pd.concat([dat_NTS,tmp_plu[['lb','ub','reject','cover']]],1)

# Coverage for OLS
dat_ols = pd.concat(holder_ols).reset_index(None,True).assign(tt='OLS')
pval_ols = norm(loc=0,scale=np.sqrt(dat_ols.Vjj)).cdf(dat_ols.bhat)
pval_ols = 2*np.minimum(pval_ols,1-pval_ols)
CI_ols = cvec(dat_ols.bhat) + np.array([-1,1])*cvec(norm.ppf(1-alpha/2)*np.sqrt(dat_ols.Vjj))
dat_ols = dat_ols.assign(reject=pval_ols<alpha,cover=np.sign(CI_ols).sum(1)==0)

# Is truncated Gaussian?
dat_TN = pd.concat(holder_SI).assign(tt='TN')
dat_TN = dat_TN.assign(abhat=lambda x: x.bhat.abs(),
                       sbhat=lambda x: np.sign(x.bhat).astype(int))
dat_TN = dat_TN.sort_values('abhat',ascending=False).reset_index(None,True)
dist_TN_ols = tnorm(mu=0,sig2=dat_TN.Vjj.values,a=cutoff,b=np.inf)
pval_TN = dist_TN_ols.cdf(dat_TN.abhat)
pval_TN = 2*np.minimum(pval_TN,1-pval_TN)
dat_TN['pval'] = pval_TN
lb_TN = dist_TN_ols.CI(x=dat_TN.abhat.values,gamma=1-alpha/2,verbose=True,tol=1e-2)
ub_TN = dist_TN_ols.CI(x=dat_TN.abhat.values,gamma=alpha/2,verbose=True,tol=1e-2)
CI_TN = np.c_[lb_TN, ub_TN]
dat_TN = dat_TN.assign(reject=pval_TN<0.05, cover=np.sign(CI_TN).sum(1)==0)

cn_sub = ['sim','tt','reject','cover']
# Save data for later
dat_bhat = pd.concat([dat_TN[cn_sub], dat_ols[cn_sub],dat_NTS_inf[cn_sub]],0).reset_index(None,True)
dat_bhat.to_csv('dat_bhat.csv',index=False)
# dat_bhat.groupby('tt')['reject','cover'].mean()

res_NTS = dat_bhat.melt('tt',['reject','cover'],'msr').groupby(['tt','msr'])
res_NTS = res_NTS.value.apply(lambda x: pd.DataFrame({'p':x.mean(),'s':x.sum(),'n':len(x)},index=[0])).reset_index()
res_NTS = pd.concat([res_NTS,pd.concat(propCI(res_NTS.s,res_NTS.n,alpha,'beta'),1)],1)
res_NTS = res_NTS.rename(columns={0:'lb',1:'ub'}).drop(columns='level_2')
hlines = pd.DataFrame({'msr':['cover','reject'],'val':[1-alpha,alpha]})
gtit = 'Vertical lines show simulation uncertainty ranges\nHorizontal lines show level target'
gg_ols = (ggplot(res_NTS,aes(x='tt',y='p')) + theme_bw() + 
    geom_point() + 
    labs(y='Percentage',x='Method') + 
    facet_wrap('~msr',scales='free_y',
        labeller=labeller(msr={'cover':'CI-Coverage','reject':'P-value < 5%'})) + 
    theme(subplots_adjust={'wspace': 0.15}) + 
    geom_linerange(aes(ymin='lb',ymax='ub')) +  ggtitle(gtit) + 
    geom_hline(aes(yintercept='val'),data=hlines,linetype='--'))
gg_ols.save(os.path.join(dir_figures,'gg_ols.png'),width=9,height=4)
