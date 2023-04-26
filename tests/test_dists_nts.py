"""
Make sure SNTN works as expected

python3 -m tests.test_dists_nts
python3 -m pytest tests/test_dists_nts.py -s

"""

# Internal
import pytest
import numpy as np
# External
from sntn.dists import nts
from parameters import seed

# Used for pytest
params_shape = [((1,)), ((10, )), ((10, 5)), ((10, 5, 2)),]


def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates NTS parameters"""
    np.random.seed(seed)
    mu1 = np.random.randn(*shape)
    mu2 = np.random.randn(*shape)
    tau1 = np.exp(np.random.randn(*shape))
    tau2 = np.exp(np.random.randn(*shape))
    a = np.random.randn(*shape)
    b = a + 1 + 2*np.random.rand(*shape)
    c1 = np.random.rand(*shape)
    c2 = 1 - c1
    return mu1, tau1, mu2, tau2, a, b, c1, c2


@pytest.mark.parametrize("shape", params_shape)
def test_nts_pdf(shape:tuple, tol:float=2e-3) -> None:
    # Draw different distributions
    mu1, tau1, mu2, tau2, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau1, mu2, tau2, a, b, c1, c2)
    # Check that numerical integration matches theory for mean
    data = dist.rvs(10000, seed)
    dmin = np.min(data, 0) - 2 
    dmax = np.max(data, 0) + 2
    points = np.linspace(dmin, dmax, 25000)
    dpoints = np.expand_dims((points[1] - points[0]).reshape(shape), 0)    
    # The pdf should integrate to close to one
    f = dist.pdf(points)
    probs = f * dpoints
    total_prob = np.sum(probs, 0)
    assert np.all(np.abs(total_prob - 1) < tol), 'Integral should sum to 1!'
    # Mean should match...
    np.sum(f * points * dpoints, 0)
    # Median should match...
    points[np.where(np.cumsum(probs, 0).round(3) == 0.5)[0].min()]
    


    
    eps = 1e-3
    start = np.minimum(dist.dist_Z1.ppf(eps) - dist.dist_Z2.ppf(eps), a.flatten())
    end = np.maximum(dist.dist_Z1.ppf(1-eps) + dist.dist_Z2.ppf(1-eps), b.flatten())
    points = np.linspace(start-12, end+12, 25000)
    dpoints = np.expand_dims((points[1] - points[0]).reshape(shape), 0)
    # The pdf should integrate to close to one
    f = dist.pdf(points)
    total_prob = np.sum(f * dpoints,0)
    assert np.all(np.abs(total_prob - 1) < tol), 'Integral should sum to 1!'
    # And the mean should matcch too
    mu_theory = dist.mean()
    
    breakpoint()
    # Cumulative probability at median should match
    
    from scipy.stats import norm
    qq = np.linspace(-4, 0, 10000)
    np.sum(norm().pdf(qq) * (qq[1] - qq[0]))
    zz = np.linspace(-4, 5, 10000)
    np.sum(norm(loc=1).pdf(zz) * (zz[1] - zz[0]) * zz)

    # idx_med = np.argmin((np.cumsum(f * dpoints, axis=0) - 0.5)**2,axis=0, keepdims=True)



@pytest.mark.parametrize("shape", params_shape)
def test_nts_rvs(shape:tuple, nsim:int=10000, tol:float=1e-1) -> None:
    # Draw different distributions
    mu1, tau1, mu2, tau2, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau1, mu2, tau2, a, b, c1, c2)
    data = dist.rvs(nsim, seed)
    # Check that the mean of the rvs aligns with E[]
    mu_rvs = data.mean(0)
    mu_theory = dist.mean()
    assert np.all(np.abs(mu_rvs - mu_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'
    


if __name__ == "__main__":
    print('--- test_nts_rvs ---')
    test_nts_rvs()


    print('~~~ The test_dists_nts.py script worked successfully ~~~')