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
    makes M <input, output> distribution pairs.
    each is sampled eta times.
    """
    def __init__(self, M=100, eta=100, verbose=True):

        self.M = M
        self.eta = eta
        if verbose:
            print '\n >>> [verbose] Making new toy data with M =',M,', eta =',eta,'\n'
        
        # mu_1, mu_2 ~ Unif[0, 1]
        self.mu_1 = np.random.rand()
        self.mu_2 = np.random.rand()

        # sig_1, sig_2 ~ Unif[0.05, 0.10]
        self.sig_1 = .05*(np.random.rand() + 1)
        self.sig_2 = .05*(np.random.rand() + 1)

    def print_params(self):
        print
        print ' >>> [debug] mu_1:',self.mu_1
        print ' >>> [debug] mu_2:',self.mu_2
        print ' >>> [debug] sig_1:',self.sig_1
        print ' >>> [debug] sig_2:',self.sig_2
        print


"""
Built-in tests.
"""
if __name__ == '__main__':

    data = toyData()
    data.print_params()
