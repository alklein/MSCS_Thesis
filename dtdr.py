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

from random import *
from multiprocessing import Pool

""" Custom Imports """
import toy
import manage_files as manager


"""
Indices for basis functions in 6D.
There will be num_terms^6 index vectors.
"""
def alphas(num_terms):
    return [[a,b,c,d,e,f] for a in range(num_terms) 
            for b in range(num_terms)
            for c in range(num_terms)
            for d in range(num_terms)
            for e in range(num_terms)
            for f in range(num_terms)]

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
List of all fourier coefficients corresponding to a given sample
fit to num_terms. Coefficients are listed in the same order as
defined by alphas().
"""
def fourier_coeffs_6D(sample, num_terms):
    indices = alphas(num_terms)
    result = []
    for alpha in indices:
        phi_alpha = cosine_basis_6D(alpha)
        coeff = np.average([phi_alpha(s) for s in sample])
        result.append(coeff)
    return result


def demo():
    mini_data = manager.load_floats('sims/sim1_exact_10.txt')
    coeffs = fourier_coeffs_6D(mini_data, 3)
    print coeffs

"""
Runs demo.
"""
if __name__ == '__main__':

    print
    print ' > RUNNING DEMO'
    demo()

