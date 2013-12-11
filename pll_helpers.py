#!/usr/bin/python2.7

"""
@file pll_helpers.py
@brief tools for parallelizing coefficient computation
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

"""
chunk generator for parallel processing
"""
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
   
"""
simple partitioning function for parallel processing
"""         
def partitioned(data, num_processes):
    return list(chunks(data, len(data) / num_processes))


"""
partitioning function for parallel error calculation
"""         
def multi_partitioned(E, test_Xs, test_Ys, KNN, k, num_processes):
    partitioned_Xs = partitioned(test_Xs, num_processes)
    partitioned_Ys = partitioned(test_Ys, num_processes)
    return [[E, partitioned_Xs[i], partitioned_Ys[i], KNN, k] for i in range(len(partitioned_Xs))]

"""
Map function for parallel computation of Fourier coeffs
"""
def coeff_Map((X, num_terms)):
    return [fourier_coeffs(x, num_terms) for x in X]
    #num_terms = packed_args[0][1]
    #return [fourier_coeffs(packed_args[i][0], num_terms) for i in range(len(packed_args))]

def meta_coeff_Map(num_terms):

    def coeff_Map(X):
        return [fourier_coeffs(x, num_terms) for x in X]

    return coeff_Map
        

class coeff_Mapper():

    def __init__(self, num_terms):
        self.num_terms = num_terms

    def coeff_Map(X):
        return [fourier_coeffs(x, num_terms) for x in X]

"""
Map function for parallel computation of test errors
"""
def err_Map(X):
    (E, test_Xs, test_Ys, KNN, k) = X
    test_X_hats = [E.nonparametric_estimation(sample, E.num_terms) for sample in test_Xs]
    test_Y_hats = [E.nonparametric_estimation(sample, E.num_terms) for sample in test_Ys]
    errs = []
    for i in range(len(test_X_hats)):
        if (KNN): est_Y_hat = E.KNN_regress(test_X_hats[i], k=k)
        else: est_Y_hat = E.regress(test_X_hats[i])
        err = E.dist_fn(test_Y_hats[i], est_Y_hat)
        errs.append(err)
    return errs
