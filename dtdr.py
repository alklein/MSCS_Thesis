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

#from pylab import *
from random import *
from sklearn import *
from multiprocessing import Pool

""" Custom Imports """
import toy
import manage_files as manager


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

class Estimator:

    """
    training_sample is a list of training instances;
    cv_sample is a list of "holdout" instances for cross-validation;
    each instance is a tuple of the form [in, out]_i.
    """
    def __init__(self, training_sample, cv_sample, num_terms = 20, dist_fn = L2_distance, kernel = RBF_kernel, nonparametric_estimation = fourier_coeffs, bandwidths = [.15, .25]): # bandwidths = [.15, .25, .5, .75, 1., 1.25, 1.5]): TEMP!!!! # NOTE: bandwidths should be tried in increasing order.

        self.num_terms = num_terms
        self.dist_fn = dist_fn
        self.kernel = kernel        
        self.nonparametric_estimation = nonparametric_estimation 
        self.bandwidths = bandwidths
        self.best_b = None

        self.Xs = training_sample[:,0]
        self.Ys = training_sample[:,1]
        self.cv_Xs = cv_sample[:,0]
        self.cv_Ys = cv_sample[:,1]

        self.ball_tree = None

    def train(self, parallel=False, num_processes=5):

        # fit training and cv data via nonparametric density estimation
        if (not parallel):            
            print ' >>> [debug] Fitting training data sequentially... '
            self.X_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.Xs]
            self.Y_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.Ys]
            print ' >>> [debug] Fitting cv data sequentially... '
            self.cv_X_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.cv_Xs]
            self.cv_Y_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.cv_Ys]
        else:
            P = Pool(processes=num_processes,)
            print ' >>> [debug] Fitting training data in parallel with',num_processes,'processes... '
            M = meta_coeff_Map(self.num_terms)

            #packed_args = [(x, self.num_terms) for x in partitioned(self.Xs, num_processes)]
            #packed_args = (partitioned(self.Xs, num_processes), self.num_terms)
            coeffs = P.map(M, partitioned(self.Xs, num_processes))
            self.X_hats = list(itertools.chain(*coeffs))

            #packed_args = [(x, self.num_terms) for x in partitioned(self.Ys, num_processes)]
            #packed_args = (partitioned(self.Ys, num_processes), self.num_terms)
            coeffs = P.map(M, partitioned(self.Ys, num_processes))
            self.Y_hats = list(itertools.chain(*coeffs))

            print ' >>> [debug] Fitting cv data in parallel with',num_processes,'processes... '
            #packed_args = [(x, self.num_terms) for x in partitioned(self.cv_Xs, num_processes)]
            #packed_args = (partitioned(self.cv_Xs, num_processes), self.num_terms)
            coeffs = P.map(M, partitioned(self.cv_Xs, num_processes))
            self.cv_X_hats = list(itertools.chain(*coeffs))

            #packed_args = [(x, self.num_terms) for x in partitioned(self.cv_Ys, num_processes)]
            #packed_args = (partitioned(self.cv_Ys, num_processes), self.num_terms)
            coeffs = P.map(M, partitioned(self.cv_Ys, num_processes))
            self.cv_Y_hats = list(itertools.chain(*coeffs))

        # cross-validate bandwidths
        print ' >>> [debug] cross-validating bandwidths...'
        b_errs = []
        for b in self.bandwidths:
            net_err = 0.
            for i in range(len(self.cv_Xs)):
                input_coeffs = self.cv_X_hats[i]
                target_coeffs = self.cv_Y_hats[i]
                Y0_coeffs = self.full_regress(input_coeffs, b=b)
                net_err += L2_distance(target_coeffs, Y0_coeffs)
            avg_err = net_err / (1.*len(self.cv_Xs))
            print ' >>> >>> [debug] Average L2 error for bandwidth',b,'-',avg_err 
            b_errs.append(avg_err)

        print ' >>> >>> [debug] Bandwidth selected:', self.bandwidths[np.argmin(b_errs)]
        self.best_b = self.bandwidths[np.argmin(b_errs)]
        if (self.best_b == self.bandwidths[0]):
            print ' >>> >>> [debug] WARNING: minimum bandwidth selected. Consider trying smaller bandwidths.'
        if (self.best_b == self.bandwidths[-1]):
            print ' >>> >>> [debug] WARNING: maximum bandwidth selected. Consider trying larger bandwidths.'    

    """
    given coeffs fit to some input sample_0, estimates the expected output distribution using all the training data.
    returns estimator in function form and, if in L2 mode, coefficient form as well.
    (in L1 mode, the coefficient form is None.)
    """
    def full_regress(self, f0, b = None):

        if (not b): b = self.best_b # possibly still None

        normed_distances = np.array([self.dist_fn(f0, f) for f in self.X_hats]) / b
        k_sum = sum([self.kernel(d) for d in normed_distances])        
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(len(self.X_hats))]

        a = np.matrix.transpose(np.array(self.Y_hats))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs


    """
    Constructs ball tree for KNN regression.
    """
    def build_ball_tree(self):
        self.ball_tree = neighbors.BallTree(self.X_hats)

    """
    like regress(), but only considers K nearest neighbors to f0
    from the training data.
    """
    def KNN_regress(self, f0, b = None, k = 1):

        if (not b): b = self.best_b # possibly still None

        distances, indices = self.ball_tree.query(f0, k=k)
        distances = distances[0]
        indices = indices[0]
        normed_distances = np.array(distances) / b
        k_sum = sum([self.kernel(d) for d in normed_distances])  
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(k)]
        selected_Ys = [self.Y_hats[index] for index in indices]

        """
        normed_distances = np.array([self.dist_fn(f0, f) for f in self.X_hats]) / b
        sorted_Xs = np.array(self.X_hats)[normed_distances.argsort()][:k]
        sorted_Ys = np.array(self.Y_hats)[normed_distances.argsort()][:k]
        normed_distances = sorted(normed_distances)[:k]
        k_sum = sum([self.kernel(d) for d in normed_distances])        
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(len(sorted_Xs))]
        """

        a = np.matrix.transpose(np.array(selected_Ys))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs


def KNN_tests():

    print ' >>> STARTING KNN TESTS <<<'
    print

    M, eta = 100, 100 # TEMP
    K = 10
    Ts = [1, 2, 3, 4, 5]

    print ' >>> Making',M,'samples...'
    tD = toy.toyData(M = M, eta = eta)
    tD.make_samples()

    all_data = tD.all_samples
    train_data = tD.train_samples
    cv_data = tD.cv_samples
    test_data = tD.test_samples

    print ' >>> Total number of toy data instances:', len(all_data)
    print ' >>> Number of training instances:', len(train_data)
    print ' >>> Number of cv instances:', len(cv_data)
    print ' >>> Number of test instances:', len(test_data)

    for T in Ts:

        print
        print ' >>> T value:',T
        print ' >>> Training estimator... '
        start = time.clock()
        E = Estimator(train_data, cv_data, num_terms = T, dist_fn = L2_distance, kernel = RBF_kernel)
        E.train(parallel=False)
        print ' >>> Train time:', time.clock() - start

        print ' >>> Building ball tree... '
        start = time.clock()
        E.build_ball_tree()
        print ' >>> Build time:', time.clock() - start
        
        print ' >>> Performing KNN regression...'
        start = time.clock()
        for i in range(len(test_data)):
            X0_sample, Y0_sample = test_data[i][0], test_data[i][1]
            X0_coeffs = fourier_coeffs(X0_sample, num_terms=T)
            'number of test coeffs:',len(X0_coeffs)
            Y0_coeffs = E.KNN_regress(X0_coeffs, k=K)
        print ' >>> KNN regression time:', time.clock() - start

        print ' >>> Performing full regression...'
        start = time.clock()
        for i in range(len(test_data)):            
            X0_sample, Y0_sample = test_data[i][0], test_data[i][1]
            X0_coeffs = fourier_coeffs(X0_sample, num_terms=T)
            Y0_coeffs = E.full_regress(X0_coeffs)
        print ' >>> Full regression time:', time.clock() - start

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

def demo():

    data = [[1, 2], [1, 3], [4, 5]]
    B = neighbors.BallTree(data)
    show()

def demo():

    data = [[1, 2], [1, 3], [4, 5]]
    B = neighbors.BallTree(data)
    show()

# TODO: implement once binning can isolate actual contiguous
# regions of simulation
def T_tests():
    pass

def tests():
    KNN_tests()
    #coeff_tests()
    #T_tests()

def demo():
    mini_data = manager.load_floats('sims/sim1_approx_1000.txt')[:100]
    print 'min x:',min(mini_data[:,0])
    print 'max x:',max(mini_data[:,0])
    print 'min vx:',min(mini_data[:,3])
    print 'max vx:',max(mini_data[:,3])

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
