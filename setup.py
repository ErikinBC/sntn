import pathlib
from setuptools import setup

dir_here = pathlib.Path(__file__).parent
README = (dir_here / 'readme.md').read_text()

setup(
    name='sntn',
    version='0.0.1',    
    description='SNTN (Sum of a Normal & Truncated Normal)',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/ErikinBC/SNTN',
    author='Erik Drysdale',
    author_email='erikinwest@gmail.com',
    license='MIT',
    packages=['sntn'],
    package_data={'sntn': ['utils','utils/*','tutorials','tutorials/*']},
    include_package_data=True,
    install_requires=['numpy', 'pandas'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
