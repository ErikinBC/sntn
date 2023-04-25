"""
Make sure SNTN works as expected

python3 -m tests.test_dists_nts
python3 -m pytest tests/test_dists_nts.py -s

"""

# Internal
import numpy as np
# External
from sntn.dists import nts

def extract_kwargs(**kwargs):
    args = []
    if 'a' in kwargs:
        a = kwargs['a']
        args.append(a)
    kwargs = {k:v for k,v in kwargs.items() if k not in ['a']}
    args = tuple(args)
    if len(args) == 0:
        return kwargs
    else:
        return *args, kwargs

a, kwargs = extract_kwargs(a=1, b=2, c=3)
a; kwargs

kwargs = extract_kwargs(b=2, c=3)
kwargs


def test_nts_() -> None:
    mu1 = np.repeat(range(4),3).reshape([4,3])
    mu2, tau1, tau2, a, b = 2, 3, 4, -1, 6
    dist = nts(mu1, mu2, tau1, tau2, a, b)
    


# CHECK THAT w1*norm.ppf()+w2*tnorm.ppf() != SNTN.ppf()


if __name__ == "__main__":
    print('--- test_nts_ ---')
    test_nts_()


    print('~~~ The test_dists_nts.py script worked successfully ~~~')