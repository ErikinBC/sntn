import pathlib
from setuptools import setup

dir_here = pathlib.Path(__file__).parent
README = (dir_here / 'readme.md').read_text()

setup(
    name='sntn',
    version='0.0.4',    
    description='Sum of a normal and a truncated Normal (SNTN)',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/ErikinBC/SNTN',
    author='Erik Drysdale',
    author_email='erikinwest@gmail.com',
    license='GPLv3',
    license_files = ('LICENSE.txt'),
    packages=['sntn'],
    package_data={'sntn': ['/*','examples/*','tests/*', 'simulations/*']},
    include_package_data=True,
    # install_requires=['numpy<=1.24.2', 'pandas<=2.0.0', 'glmnet<=2.2.1', 'scipy<=1.9.3'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
