"""
Hard coded parameters, used for package dev
"""

from sntn.utilities.utils import makeifnot

# Folder set up
dir_figures = 'figures'
makeifnot(dir_figures)
dir_simulations = 'simulations'
makeifnot(dir_simulations)

# Reproducability seed in unittesting
seed = 1