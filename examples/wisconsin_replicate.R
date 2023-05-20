# Load modules
library(selectiveInference)
alpha = 0.05

# Load data & normalize
path_wisconsin = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/breast_cancer.csv'
Xy = read.csv(path_wisconsin, skip = 1, header = FALSE)
y = Xy[,ncol(Xy)]
X = as.matrix(Xy[,-ncol(Xy)])
n = nrow(X)
p = ncol(X)
cn = c('mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error','texture error','perimeter error','area error','smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension')
colnames(X) = cn
xmu = apply(X, 2, mean)
xse = apply(X, 2, sd)
xtil = sweep(sweep(X, 2, xmu, '-'), 2, xse, '/')
ytil = y - mean(y)

# Run Lasso
lamb_glmnet = 0.1
sigma = 1
bhat_lasso = coef(glmnet(x=xtil, y=ytil, alpha=1, standardize=FALSE, lambda=lamb_glmnet))
bhat_lasso = as.vector(bhat_lasso)[-1]
posi = fixedLassoInf(x=xtil, y=ytil, beta=bhat_lasso, lambda=lamb_glmnet*n, alpha=alpha, sigma=sigma)
round(data.frame(Var=posi$vars,Coef=posi$coef0, `P-value`=posi$pv, lb=posi$ci[,1], ub=posi$ci[,2]),3)
