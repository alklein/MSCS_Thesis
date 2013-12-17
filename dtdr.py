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
import constants
import manage_files as manager

from math_helpers import *
from pll_helpers import *


class ONE_D_Estimator:

    """
    training_sample is a list of training instances;
    cv_sample is a list of "holdout" instances for cross-validation;
    each instance is a tuple of the form [in, out]_i.
    """
    # bandwidths = [.15, .25, .5, .75, 1., 1.25, 1.5]): TEMP!!!! # NOTE: bandwidths should be tried in increasing order.
    def __init__(self, training_sample, cv_sample, num_terms = 20, dist_fn = L2_distance, kernel = RBF_kernel, nonparametric_estimation = fourier_coeffs, bandwidths = [.15, .25]): 

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

            coeffs = P.map(M, partitioned(self.Xs, num_processes))
            self.X_hats = list(itertools.chain(*coeffs))

            coeffs = P.map(M, partitioned(self.Ys, num_processes))
            self.Y_hats = list(itertools.chain(*coeffs))

            print ' >>> [debug] Fitting cv data in parallel with',num_processes,'processes... '
            coeffs = P.map(M, partitioned(self.cv_Xs, num_processes))
            self.cv_X_hats = list(itertools.chain(*coeffs))

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

        a = np.matrix.transpose(np.array(selected_Ys))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs


class ND_Estimator:

    """
    training_sample is a list of training instances;
    cv_sample is a list of "holdout" instances for cross-validation;
    each instance is a tuple of the form [in, out]_i.
    """
    def __init__(self, train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = 20, dim = 6, dist_fn = L2_distance, kernel = RBF_kernel, nonparametric_estimation = fourier_coeffs_ND, bandwidths = [.15, .25]): # bandwidths = [.15, .25, .5, .75, 1., 1.25, 1.5]): TEMP!!!! # NOTE: bandwidths should be tried in increasing order.

        self.degree = degree
        self.dim = dim
        self.dist_fn = dist_fn
        self.kernel = kernel        
        self.nonparametric_estimation = nonparametric_estimation 
        self.bandwidths = bandwidths
        self.best_b = None

        self.Xs = train_samples_in
        self.Ys = train_samples_out
        self.cv_Xs = cv_samples_in
        self.cv_Ys = cv_samples_out
        
        self.ball_tree = None

    def train(self, parallel=False, num_processes=5):

        # fit training and cv data via nonparametric density estimation
        if (not parallel):            
            print ' >>> [debug] Fitting training data sequentially... '
            self.X_hats = [self.nonparametric_estimation(sample, self.degree, self.dim) for sample in self.Xs]
            self.Y_hats = [self.nonparametric_estimation(sample, self.degree, self.dim) for sample in self.Ys]
            print ' >>> [debug] Fitting cv data sequentially... '
            self.cv_X_hats = [self.nonparametric_estimation(sample, self.degree, self.dim) for sample in self.cv_Xs]
            self.cv_Y_hats = [self.nonparametric_estimation(sample, self.degree, self.dim) for sample in self.cv_Ys]
        else:
            P = Pool(processes=num_processes,)
            print ' >>> [debug] Fitting training data in parallel with',num_processes,'processes... '
            M = meta_coeff_Map(self.degree)

            coeffs = P.map(M, partitioned(self.Xs, num_processes))
            self.X_hats = list(itertools.chain(*coeffs))

            coeffs = P.map(M, partitioned(self.Ys, num_processes))
            self.Y_hats = list(itertools.chain(*coeffs))

            print ' >>> [debug] Fitting cv data in parallel with',num_processes,'processes... '
            coeffs = P.map(M, partitioned(self.cv_Xs, num_processes))
            self.cv_X_hats = list(itertools.chain(*coeffs))

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

        a = np.matrix.transpose(np.array(selected_Ys))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs

def test_errs_3D(E, test_Xs, test_Ys):
    test_X_hats = [E.nonparametric_estimation(sample, E.degree, E.dim) for sample in test_Xs]
    test_Y_hats = [E.nonparametric_estimation(sample, E.degree, E.dim) for sample in test_Ys]
    errs = []
    for i in range(len(test_X_hats)):
        est_Y_hat = E.full_regress(test_X_hats[i])
        err = E.dist_fn(test_Y_hats[i], est_Y_hat)
        errs.append(err)
    return errs

def KNN_tests_1D():

    print ' >>> STARTING 1D KNN TESTS <<<'
    print

    M, eta = 1000, 1000
    K = 5
    Ts = range(1, 21)

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

    ball_build_times = []
    KNN_regress_times = []
    full_regress_times = []

    for T in Ts:

        print
        print ' >>> T value:',T
        print ' >>> Training estimator... '
        start = time.clock()
        E = ONE_D_Estimator(train_data, cv_data, num_terms = T, dist_fn = L2_distance, kernel = RBF_kernel)
        E.train(parallel=False)
        print ' >>> Train time:', time.clock() - start

        print ' >>> Building ball tree... '
        start = time.clock()
        E.build_ball_tree()
        ball_build_times.append(time.clock() - start)
        print ' >>> Build time:', ball_build_times[-1]
        
        print ' >>> Performing KNN regression...'
        start = time.clock()
        for i in range(len(test_data)):
            X0_sample, Y0_sample = test_data[i][0], test_data[i][1]
            X0_coeffs = fourier_coeffs(X0_sample, num_terms=T)
            'number of test coeffs:',len(X0_coeffs)
            Y0_coeffs = E.KNN_regress(X0_coeffs, k=K)
        KNN_regress_times.append(time.clock() - start)
        print ' >>> KNN regression time:', KNN_regress_times[-1]

        print ' >>> Performing full regression...'
        start = time.clock()
        for i in range(len(test_data)):            
            X0_sample, Y0_sample = test_data[i][0], test_data[i][1]
            X0_coeffs = fourier_coeffs(X0_sample, num_terms=T)
            Y0_coeffs = E.full_regress(X0_coeffs)
        full_regress_times.append(time.clock() - start)
        print ' >>> Full regression time:', full_regress_times[-1]

    print
    print ' >>> Experiments complete.'
    print ' >>> Ball build times:', ball_build_times
    print ' >>> KNN regress times:', KNN_regress_times
    print ' >>> Full regress times:', full_regress_times

def KNN_tests_ND(dim=3):

    print ' >>> STARTING KNN TESTS IN',dim,'DIMS <<<'

    Ts = range(1, 11)
    M, eta = 5000, 5000
    K = 10

    print
    print ' >>> Extracting data with M =',M,' eta =',eta,' dim =',dim
    data = manager.load_partial('sim1_exact.txt', M, dim, 15)

    print
    print ' >>> Length of data:', len(data)
    print ' >>> Number of samples per bin:',len(data[0])
    print ' >>> Dimensionality of each sample:',len(data[0][0])

    print 
    print ' >>> Before scaling,'
    print ' >>> >>> min max col 0:', manager.col_min_max(data, 0)
    print ' >>> >>> min max col 1:', manager.col_min_max(data, 1)
    for i in range(dim):
        data = manager.scale_col_emp(data, i)
    print ' >>> After scaling,'
    print ' >>> >>> min max col 0:', manager.col_min_max(data, 0)
    print ' >>> >>> min max col 1:', manager.col_min_max(data, 1)

    [train_samples, cv_samples, test_samples] = manager.partition_data(data)
    print
    print ' >>> Number of train samples:',len(train_samples)
    print ' >>> Number of cv samples:',len(cv_samples)

    train_samples_in = train_samples
    train_samples_out = train_samples
    cv_samples_in = cv_samples
    cv_samples_out = cv_samples
    test_samples_in = test_samples
    test_samples_out = test_samples

    ball_build_times = []
    KNN_regress_times = []
    full_regress_times = []

    # TEMP
    print train_samples_in

    for T in Ts:

        print
        print ' >>> T value:',T
        print ' >>> Training estimator... '
        start = time.clock()
        E = ND_Estimator(train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = T, dim = dim)
        E.train(parallel=False)
        print ' >>> Train time:', time.clock() - start

        print ' >>> Building ball tree... '
        start = time.clock()
        E.build_ball_tree()
        ball_build_times.append(time.clock() - start)
        print ' >>> Build time:', ball_build_times[-1]
        
        print ' >>> Performing KNN regression...'
        start = time.clock()
        for i in range(len(test_samples)):
            X0_sample, Y0_sample = test_samples_in[i], test_samples_out[i]
            X0_coeffs = fourier_coeffs_ND(X0_sample, degree=T, dim=dim)
            'number of test coeffs:',len(X0_coeffs)
            Y0_coeffs = E.KNN_regress(X0_coeffs, k=K)
        KNN_regress_times.append(time.clock() - start)
        print ' >>> KNN regression time:', KNN_regress_times[-1]

        print ' >>> Performing full regression...'
        start = time.clock()
        for i in range(len(test_samples)):            
            X0_sample, Y0_sample = test_samples_in[i], test_samples_out[i]
            X0_coeffs = fourier_coeffs_ND(X0_sample, degree=T, dim=dim)
            Y0_coeffs = E.full_regress(X0_coeffs)
        full_regress_times.append(time.clock() - start)
        print ' >>> Full regression time:', full_regress_times[-1]

    print
    print ' >>> Experiments complete.'
    print ' >>> Ball build times:', ball_build_times
    print ' >>> KNN regress times:', KNN_regress_times
    print ' >>> Full regress times:', full_regress_times
    

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

def bin_tests():    

    #num_bins = int(32768**(1./3))
    num_bins = 5
    print 'num bins:',num_bins

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/num_bins
    binsz_y = (ymax - ymin)/num_bins
    binsz_z = (zmax - zmin)/num_bins

    print
    print 'binsz_x:',binsz_x
    print 'binsz_y:',binsz_y
    print 'binsz_z:',binsz_z

    bindices = manager.bindices_3D(num_bins) #[:100] # TEMP: consider subset of bins
    print
    print 'bindices:'
    print bindices

    manager.count_particles_3D('sims/new_sim1_exact.txt', bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=True)

#    assignments = manager.assign_particles_3D('sims/new_sim1_exact.txt', bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=True)
#    np.savetxt('assign.txt', [])
#    manager.save_assignments_3D(assignments, 5, 'assign.txt', xmin, xmax, ymin, ymax, zmin, zmax)    

    #bindex = [0, 0, 0]
    #ps = manager.load_bin_3D('sims/sim1_exact.txt', bindex, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, verbose=True)

def my_writetxt(filename, data):
    np.savetxt(filename, [])
    f = open(filename, 'r+')
    for line in data:
        outp = ''
        for val in line:
            outp += str(val) + ' '
        outp += '\n'
        f.write(outp)
    f.close()

def ID_tests():

    #num_bins = int(32768**(1./3))
    num_bins = 18
    bindex = [0, 0, 0]

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/num_bins
    binsz_y = (ymax - ymin)/num_bins
    binsz_z = (zmax - zmin)/num_bins

    # isolate cube; put in file
    #ps = manager.load_bin_3D('sims/new_sim1_exact.txt', bindex, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, verbose=True)
    #my_writetxt('ex_bin.txt', ps)

    # rescale 

    new_num_bins = 10

    new_xmin, new_xmax = xmin + bindex[0]*binsz_x, xmin + (bindex[0] + 1)*binsz_x
    new_ymin, new_ymax = ymin + bindex[1]*binsz_y, ymin + (bindex[1] + 1)*binsz_y
    new_zmin, new_zmax = zmin + bindex[2]*binsz_z, zmin + (bindex[2] + 1)*binsz_z

    print
    print 'new x range:',new_xmin,new_xmax
    print 'new y range:',new_ymin,new_ymax
    print 'new z range:',new_zmin,new_zmax

    new_binsz_x = (new_xmax - new_xmin)/new_num_bins
    new_binsz_y = (new_ymax - new_ymin)/new_num_bins
    new_binsz_z = (new_zmax - new_zmin)/new_num_bins

    print
    print 'new binsz_x:',new_binsz_x
    print 'new binsz_y:',new_binsz_y
    print 'new binsz_z:',new_binsz_z

    new_bindices = manager.bindices_3D(new_num_bins)    

    assignments = manager.assign_particles_3D('ex_bin.txt', new_bindices, new_xmin, new_ymin, new_zmin, new_binsz_x, new_binsz_y, new_binsz_z, new_num_bins, verbose=True, chunk=10000)
    input_ps = []
    for key in assignments:
        cur = assignments[key]
        if len(cur) > 0: input_ps.append(cur)

    print
    print 'number of training instances available:', len(input_ps)

    # run algorithm with some fraction of inputs

    for i in range(3):
        input_ps = manager.scale_col_emp(input_ps, i)

    # if desired: choose some subset of the training samples here. 
    # run following tests for different sizes

    [train_samples, cv_samples, test_samples] = manager.partition_data(input_ps)

    train_samples_in = train_samples
    train_samples_out = train_samples
    cv_samples_in = cv_samples
    cv_samples_out = cv_samples
    test_samples_in = test_samples
    test_samples_out = test_samples

    T = 3
    dim = 3

    E = ND_Estimator(train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = T, dim = dim)
    E.train(parallel=False)

    # compute average test error on test samples
    avg_err = np.average(test_errs_3D(E, test_samples_in, test_samples_out))
    print
    print 'average test error:', avg_err

def misc():

    num_bins = int(32768**(1./3))
    bindex = [0, 0, 0]
    bindices = [bindex]

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/num_bins
    binsz_y = (ymax - ymin)/num_bins
    binsz_z = (zmax - zmin)/num_bins

    ass = manager.count_particles_3D('sims/new_sim1_exact.txt', bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=True)

def data_tests():

    assignments = {}
    assignments['[0, 0, 0]'] = [[1, 2, 3, 4, 5, 6], [3, 3, 3, 3, 3, 3], [4, 4, 4, 3, 3, 3]]
    assignments['[0, 1, 1]'] = [[1, 2, 3, 4, 5, 6], [3, 3, 3, 3, 3, 3], [4, 4, 4, 3, 3, 2]]
    assignments['[0, 2, 0]'] = [[1, 2, 3, 4, 5, 6], [3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2, 2]]
    
    xmin, xmax = 1., 6.
    ymin, ymax = 1., 6.
    zmin, zmax = 1., 6.

    np.savetxt('assign.txt', [])
    manager.save_assignments_3D(assignments, 3, 'assign.txt', xmin, xmax, ymin, ymax, zmin, zmax)
    coeffs = np.loadtxt('assign.txt')
    print coeffs

def demo():

    data = [[1, 2], [1, 3], [4, 5]]
    B = neighbors.BallTree(data)
    show()

# TODO: implement once binning can isolate actual contiguous
# regions of simulation
def T_tests():
    pass

def tests():
    #KNN_tests_1D()
    #KNN_tests_ND()
    #coeff_tests()
    #T_tests()
    #bin_tests()
    #misc()
    ID_tests()
    #data_tests()

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
