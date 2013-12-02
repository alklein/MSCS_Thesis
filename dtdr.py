#!/usr/bin/python2.7

"""
@file dtdr.py
@brief tools for distribution-to-distribution regression
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

from pylab import *
from random import *
from multiprocessing import Pool

""" Custom Imports """
import toy
import manage_files as manager


"""
Indices for basis functions in 6D.
There will be degree^6 index vectors.
"""
def alphas(degree):
    return [[a,b,c,d,e,f] for a in range(degree) 
            for b in range(degree)
            for c in range(degree)
            for d in range(degree)
            for e in range(degree)
            for f in range(degree)]

"""
6D basis function corresponding to a given index vector (alpha).
"""
def cosine_basis_6D(alpha):
    
    # x should be a 6D vector of the form [a, b, c, d, e, f]
    def phi_alpha(x):
        result = 1
        for i in range(6):
            phi_i = toy.cosine_basis(alpha[i])
            result *= phi_i(x[i])
        return result

    return phi_alpha

"""
List of all degree^6 fourier coefficients corresponding to a given 
sample fit to num_terms. Coefficients are listed in the same order 
as defined by alphas().
"""
def fourier_coeffs_6D(sample, degree):
    indices = alphas(degree)
    result = []
    for alpha in indices:
        phi_alpha = cosine_basis_6D(alpha)
        coeff = np.average([phi_alpha(s) for s in sample])
        result.append(coeff)
    return result

def coeff_tests():
    mini_data = manager.load_floats('sims/sim1_approx_1000.txt')[:100]
    degrees = range(6)
    times = []
    for deg in degrees:
        start = time.clock()
        coeffs = fourier_coeffs_6D(mini_data, deg)
        times.append(time.clock() - start)
        print 'degree:',deg,'  num coeffs:',len(coeffs)

    figure(0)
    plot(degrees, times, '-')
    xlabel('Degree', fontsize=24)
    ylabel('Time to Compute Coefficients (s)', fontsize=24)

    figure(1)
    semilogy(degrees, times, '-')
    xlabel('Degree', fontsize=24)
    ylabel('Time to Compute Coefficients (s)', fontsize=24)

    show()

# TODO: implement once binning can isolate actual contiguous
# regions of simulation
def T_tests():
    pass

def tests():

    coeff_tests()
    T_tests()

def demo():
    pass


"""
Runs demo.
"""
if __name__ == '__main__':

    print
    print ' > RUNNING TESTS'
    tests()

    print
    print ' > RUNNING DEMO'
    demo()
