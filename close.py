#!/usr/bin/python2.7

"""
@file close.py
@brief code used to examine simulation subsets
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

""" General Imports """
import os
import sys
import time
import math
import itertools
import numpy as np

#from pylab import *
from random import *
from sklearn import *
from multiprocessing import Pool

""" Custom Imports """
import toy
import constants
import manage_files as manager

from math_helpers import *
from pll_helpers import *
from dtdr import isolate_particles


XX, YY, ZZ, VX, VY, VZ = range(6)

"""
Creation of isolated cube data. 

18 divisions per axis -> 18**3 cubes,
17 divisions per axis -> 17**3 cubes; 
data from cube nearest the origin is retained.
"""
#isolate_particles(div_per_axis = 18, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_18.txt')
#isolate_particles(div_per_axis = 18, bindex = [0, 0, 0], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_18.txt')

#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_17.txt')
#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_17.txt')

exact_cube = np.loadtxt('sim1_partial_exact_18.txt')
approx_cube = np.loadtxt('sim1_partial_approx_18.txt')

print
print 'number of particles in exact cube:',len(exact_cube)
print 'number of particles in approx cube:',len(approx_cube)

print
print 'exact X range:',min(exact_cube[:,XX]),max(exact_cube[:,XX])
print 'approx X range:',min(approx_cube[:,XX]),max(approx_cube[:,XX])

print
print exact_cube[:10]
print
print approx_cube[:10]
