#!/usr/bin/python

"""
@file toy.py
@brief tools to make and sample toy 1-D distributions
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

import math
import numpy as np


# TODO: implement
def rejection_sample(pdf, count):
    return []

class norm_pdf_dist:
    
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig

    def eval(self, x):
        return (1./(2*math.pi*self.sig**2)**.5) * math.exp(-(x - self.mu)**2/(2*self.sig**2))

class norm_cdf_dist:
    
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig

    def eval(self, x):
        return .5 * (1 + math.erf((x - self.mu) / ((2 * self.sig**2)**.5)))

class g_dist:

    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig
        self.phi = norm_pdf_dist(mu, sig)
        self.PHI1 = norm_cdf_dist(mu, sig) 
        self.PHI2 = norm_cdf_dist(mu, sig) 

    def eval(self, x):
        return (1./self.sig) * ((self.phi.eval((x - self.mu)/self.sig)) / (self.PHI1.eval((1. - self.mu)/self.sig) - self.PHI2.eval(-self.mu/self.sig)))

class p_dist:

    def __init__(self, mu_1, mu_2, sig_1, sig_2):
        self.g1 = g_dist(mu_1, sig_1)
        self.g2 = g_dist(mu_2, sig_2)

    def eval(self, x):
        return .5 * (self.g1.eval(x) + self.g2.eval(x))

class q_dist:

    def __init__(self, mu_1, mu_2, sig_1, sig_2):
        self.g1 = g_dist(1 - mu_1, sig_1)
        self.g2 = g_dist(1 - mu_2, sig_2)

    def eval(self, x):
        return .5 * (self.g1.eval(x) + self.g2.eval(x))

class toyData:

    """
    Makes new toyData object.
    """
    def __init__(self, M=100, eta=100, verbose=True):

        self.M = M
        self.eta = eta
        self.samples = None
        self.functions = None

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
    def make_samples(self):
        samples = []
        for i in range(self.M):
            
            # mu_1, mu_2 ~ Unif[0, 1]
            mu_1 = np.random.rand()
            mu_2 = np.random.rand()
            
            # sig_1, sig_2 ~ Unif[0.05, 0.10]
            sig_1 = .05*(np.random.rand() + 1)
            sig_2 = .05*(np.random.rand() + 1)

            # create and sample probability distributions
            p = p_dist(mu_1, mu_2, sig_1, sig_2)
            q = q_dist(mu_1, mu_2, sig_1, sig_2)
            input_samples = rejection_sample(p.eval, self.eta)
            output_samples = rejection_sample(q.eval, self.eta)
            samples.append([input_samples, output_samples])

        samples = np.array(samples)
        self.samples = samples
        return samples

    def make_functions(self):
        # use self.samples
        print ' >>> TODO: implement make_functions()'

    """
    Debugging method.
    """
    def print_params(self):
        print
        print ' >>> [debug] M:',self.M
        print ' >>> [debug] eta:',self.eta
        print


"""
Demos code and runs built-in tests.
"""
if __name__ == '__main__':

    print
    print ' > RUNNING BUILT-IN TESTS'
    print
    print ' > [debug] Making new toyData object...'
    tD = toyData()
    print ' > [debug] Checking param values...'
    tD.print_params()
    print ' > [debug] Generating toy data...'
    data = tD.make_samples()
    print ' > [debug] Number of toy training instances:', len(data)
    print ' > [debug] Length of input, output pairs:', len(data[0])
    print ' > [debug] Number of samples per distribution:', len(data[0][0])
    print
