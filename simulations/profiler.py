"""
Used to run different profile checks

python3 -m simulations.profiler --func1
"""

import argparse
from line_profiler import LineProfiler

# Example function definitions
def func1():
    print("Running func1")
    for i in range(10000):
        pass

def func2():
    print("Running func2")
    import numpy as np
    dat1 = np.random.randn(100, 3)
    dat2 = np.random.randn(3, 20)
    mat = dat1.dot(dat2)

def profile_function(func):
    """Profile the specified function using LineProfiler."""
    lp = LineProfiler()
    lp_wrapper = lp(func)
    lp_wrapper()  # Run the function through LineProfiler
    lp.print_stats()  # Print the profiling results directly

def main():
    parser = argparse.ArgumentParser(description='Run profiler on different functions.')
    parser.add_argument('--func1', action='store_true', help='Profile func1')
    parser.add_argument('--func2', action='store_true', help='Profile func2')

    args = parser.parse_args()

    if args.func1:
        profile_function(func1)
    elif args.func2:
        profile_function(func2)
    else:
        print("No function specified. Please use --func1 or --func2.")

if __name__ == "__main__":
    main()