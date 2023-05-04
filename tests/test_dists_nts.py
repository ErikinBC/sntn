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
params_shape = [((1,)), ((12, )), ((4, 3)), ((3, 2, 2)),]


def gen_params(shape:tuple or list, seed:int or None) -> tuple:
    """Convenience wrapper for generates NTS parameters"""
    np.random.seed(seed)
    mu1 = np.random.randn(*shape)
    mu2 = np.random.randn(*shape)
    tau21 = np.exp(np.random.randn(*shape)) + 1
    tau22 = np.exp(np.random.randn(*shape)) + 1
    a = mu2 - np.abs(np.random.randn(*shape)) - 0.1
    b = mu2 + np.abs(np.random.rand(*shape)) + 0.1
    c1 = np.random.rand(*shape)
    c2 = 1 - c1
    return mu1, tau21, mu2, tau22, a, b, c1, c2


@pytest.mark.parametrize("shape", params_shape)
def test_nts_cdf(shape:tuple, ndraw:int=10000, tol_cdf:float=0.01) -> None:
    """Checks that:
    i) CDF aligns with classic 1964 paper
    ii) Empirical rvs aligns with cdf
    """ 
    # (i) Sanity check on 1964 query
    mu1, tau21 = 100, 6**2
    mu2, tau22 = 50, 3**2
    a, b = 44, np.inf
    w = 138
    dist_1964 = nts(mu1, tau21, mu2, tau22, a, b)
    expected_1964 = 0.03276
    cdf_1964 = dist_1964.cdf(w)[0]
    assert np.round(cdf_1964,5) == expected_1964, F'Expected CDF to be: {expected_1964} not {cdf_1964}' 
    
    # (ii) Check that random parameters align with rvs
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    x = dist.rvs(1)[0,...]
    cdf_theory = dist.cdf(x)
    cdf_rvs = np.mean(dist.rvs(ndraw, seed) <= x, 0)
    err_max = np.abs(cdf_theory - cdf_rvs).max()
    assert err_max < tol_cdf, f'Maximum error {err_max} was greater than {tol_cdf}'

    

@pytest.mark.parametrize("shape", params_shape)
def test_nts_pdf(shape:tuple, tol_cdf:float=0.005, tol_mu:float=0.1) -> None:
    """Checks that:
    i) CDF integrates to one
    ii) Mean = int f(x) x dx 
    iii) Median = inf_c : int_{lb}^c f(x) dx = 0.5
    """ 
    # Draw different distributions
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    # Check that numerical integration matches theory for mean
    data = dist.rvs(10000, seed)
    dmin = np.min(data, 0) - 1
    dmax = np.max(data, 0) + 1
    points = np.linspace(dmin, dmax, 25000)
    dpoints = np.expand_dims((points[1] - points[0]).reshape(shape), 0)    
    # The pdf should integrate to close to one
    f = dist.pdf(points)
    probs = f * dpoints
    total_prob = np.sum(probs, 0)
    assert np.all(np.abs(total_prob - 1) < tol_cdf), 'Integral should sum to 1!'
    # Mean should match...
    mu_int = np.sum(f * points * dpoints, 0)
    assert np.all(np.abs(mu_int - np.mean(data, 0)) < tol_mu), 'Means do not align'
    # Median should match...
    idx_median = np.expand_dims(np.argmin(np.cumsum(probs, 0) <= 0.5,axis=0),0)
    med_int = np.take_along_axis(points, idx_median, 0)
    med_data = np.median(data, 0, keepdims=True)
    abs_err_med = np.abs(med_int - med_data)
    assert np.all(abs_err_med < tol_mu), 'Medians do not align'


@pytest.mark.parametrize("shape", params_shape)
def test_nts_rvs(shape:tuple, nsim:int=10000, tol:float=1e-1) -> None:
    """Checks that the random variables align with expected mean"""
    # Draw different distributions
    mu1, tau21, mu2, tau22, a, b, c1, c2 = gen_params(shape, seed)
    dist = nts(mu1, tau21, mu2, tau22, a, b, c1, c2)
    data = dist.rvs(nsim, seed)
    # Check that the mean of the rvs aligns with E[]
    mu_rvs = data.mean(0)
    mu_theory = dist.mean()
    assert np.all(np.abs(mu_rvs - mu_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'
    # # Check median
    # med_rvs = np.median(data, 0)
    # med_theory = dist.ppf(0.5)
    # assert np.all(np.abs(med_rvs - med_theory) < tol),  f'Expected the actual/theoretical mean to be within {tol} of each other'
    


if __name__ == "__main__":
    print('--- test_nts_rvs ---')
    test_nts_rvs()
    test_nts_pdf()


    print('~~~ The test_dists_nts.py script worked successfully ~~~')