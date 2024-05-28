import pathlib
from setuptools import setup, find_packages

dir_here = pathlib.Path(__file__).parent
README = (dir_here / 'readme.md').read_text()

setup(
    name='sntn',
    version='0.1.0',    
    description='Sum of a normal and a truncated Normal (SNTN)',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/ErikinBC/SNTN',
    author='Erik Drysdale',
    author_email='erikinwest@gmail.com',
    license='GPLv3',
    license_files=('LICENSE.txt',),
    packages=find_packages(),  # This will find any folders with __init__.py
    package_data={
        'sntn': ['*.py', 'examples/*', 'tests/*', 'simulations/*', 'benchmark/*', 'utilities/*']
    },
    python_requires='>=3.10.0, <=3.11.9',
    install_requires=[
        'plotnine>=0.12.1',
        'numpy>=1.25.0',
        'pandas>=1.5.3',
        'scikit_learn>=1.2.2',
        'scipy>=1.10.1',
        'statsmodels>=0.14.0'
        ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
