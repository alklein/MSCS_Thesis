#!/usr/bin/python2.7

"""
@file math_helpers.py
@brief mathematical helper functions for use in distribution-to-distribution regression
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

""" General Imports """
import os
import sys
import math
import itertools
import numpy as np

from random import *
from sklearn import *

"""
returns indexed function from the cosine basis.
"""
def cosine_basis(index):

    def one(x):
        return 1

    def phi(x):
        return (2**.5) * math.cos(math.pi * index * x)

    if (not index): return one
    else: return phi

"""
c1, c2 are coefficient vectors. they should have the same length.
"""
def L2_distance(c1, c2):
    return sum([(c1[i] - c2[i])**2 for i in range(len(c1))])

"""
i.e. Gaussian kernel function;
defined for x in [0, +inf)
"""
def RBF_kernel(x):
    return math.exp(-(x**2)/2.)

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

def alphas_2D(degree):
    return [[a,b] for a in range(degree)
            for b in range(degree)]

def alphas_3D(degree):
    return [[a,b,c] for a in range(degree)
            for b in range(degree)
            for c in range(degree)]

def alphas_ND(dim):
    if (dim == 2): return alphas_2D
    elif (dim == 3): return alphas_3D
    elif (dim == 6): return alphas
    else: return None

def cosine_basis_ND(alpha, dim):

    def phi_alpha(x):
        result = 1
        for i in range(dim):
            phi_i = cosine_basis(alpha[i])
            result *= phi_i(x[i])
        return result

    return phi_alpha

"""
6D basis function corresponding to a given index vector (alpha).
"""
def cosine_basis_6D(alpha):
    
    # x should be a 6D vector of the form [a, b, c, d, e, f]
    def phi_alpha(x):
        result = 1
        for i in range(6):
            phi_i = cosine_basis(alpha[i])
            result *= phi_i(x[i])
        return result

    return phi_alpha

"""
List representation of 1D fourier coefficients corresponding to nonparametric 
estimation of the distribution from which sample was drawn.

used to compute L2 distance.
"""
def fourier_coeffs(sample, num_terms):
    result = []
    for index in range(num_terms):
        phi = cosine_basis(index)
        coeff = sum([phi(s) for s in sample])/(1.*len(sample))
        result.append(coeff)
    return result

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

"""
List of all degree^N fourier coefficients corresponding to a given 
sample fit to num_terms. Coefficients are listed in the same order 
as defined by alphas().
"""
def fourier_coeffs_ND(sample, degree, dim):

    indices = alphas_ND(dim)(degree)
    result = []
    for alpha in indices:
        phi_alpha = cosine_basis_ND(alpha, dim)
        coeff = np.average([phi_alpha(s) for s in sample])
        result.append(coeff)
    return result


