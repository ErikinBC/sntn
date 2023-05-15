library(selectiveInference)

# Set path
dir_base = "/Users/drysdaleerik/Documents/code/nts/R"
setwd(dir_base)
# Load the x/y data
dat = read.csv('tmp_xy.csv')
y = dat$y
x = as.matrix(dat[,-1])
n = length(y)

# Fit lasso
lam = 0.3154675503852345
mdl_glmnet = glmnet(x, y, 'gaussian', lambda = lam)
bhat_lasso = as.vector(mdl_glmnet$beta)
bhat_M = bhat_lasso[bhat_lasso != 0]

# Run inference (have to adjust lambda by n)
sigma2 = 1.1
browser(fixedLassoInf(x, y, bhat_lasso, n*lam, sigma=sqrt(sigma2)))
posi = fixedLassoInf(x, y, bhat_lasso, n*lam, sigma=sqrt(sigma2))

