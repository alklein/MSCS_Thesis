#!/usr/bin/python

"""
@file toy.py
@brief tools to make and sample toy 1-D distributions
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

import numpy as np


class toyData:

    """
    Makes new toyData object.
    """
    def __init__(self, M=100, eta=100, verbose=True):

        self.M = M
        self.eta = eta
        self.data = None

        if verbose:
            print '\n >>> [verbose] New toyData object made with M =',str(M) + ',','eta =',eta,'\n'
        
    """
    Makes M <input, output> distribution pairs. 
    Each is sampled eta times.

    Training instance i uses input values drawn from the pdf p_i
    and output values drawn from the pdf q_i.

    Returns a length-M list of training instances; 
    each instance is a tuple of the form [in, out]_i,     
    where in and out are length-eta lists of scalar values.
    """
    def make_data(self):
        data = []
        for i in range(self.M):
            
            # mu_1, mu_2 ~ Unif[0, 1]
            mu_1 = np.random.rand()
            mu_2 = np.random.rand()
            
            # sig_1, sig_2 ~ Unif[0.05, 0.10]
            sig_1 = .05*(np.random.rand() + 1)
            sig_2 = .05*(np.random.rand() + 1)

            # TODO: create and append new instance

        data = np.array(data)
        self.data = data
        return data

    """
    Debugging method.
    """
    def print_params(self):
        print
        print ' >>> [debug] M:',self.M
        print ' >>> [debug] eta:',self.eta
        print


"""
Defaults to running built-in tests.
"""
if __name__ == '__main__':

    print
    print ' > RUNNING BUILT-IN TESTS'
    print
    print ' > [debug] Making new toyData object'
    tD = toyData()
    print ' > [debug] Checking param values'
    tD.print_params()
    print ' > [debug] Generating toy data'
    data = tD.make_data()
    print ' > [debug] Length of toy data:',len(data)
    print
