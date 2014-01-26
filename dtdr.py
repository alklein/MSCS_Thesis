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
    def __init__(self, train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = 20, dim = 6, dist_fn = L2_distance, kernel = RBF_kernel, nonparametric_estimation = fourier_coeffs_ND, bandwidths = [.5, 1, 2, 5]): # bandwidths = [.15, .25, .5, .75, 1., 1.25, 1.5]): TEMP!!!! # NOTE: bandwidths should be tried in increasing order.

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
        if (k_sum == 0): 
            print ' >>> >>> [debug] WARNING: k_sum underflow detected'
            k_sum = .00001
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

"""
Given an estimator E, input samples (test_Xs), and output samples (test_Ys),
nonparametrically estimates the samples and performs the regression;
returns the vector of errors.
"""
def test_errs_3D(E, test_Xs, test_Ys):
    test_X_hats = [E.nonparametric_estimation(sample, E.degree, E.dim) for sample in test_Xs]
    test_Y_hats = [E.nonparametric_estimation(sample, E.degree, E.dim) for sample in test_Ys]
    errs = []
    for i in range(len(test_X_hats)):
        est_Y_hat = E.full_regress(test_X_hats[i])
        err = E.dist_fn(test_Y_hats[i], est_Y_hat)
        errs.append(err)
    return errs

"""
Custom function, like numpy's savetxt(), to forcibly write data to file.
"""
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

"""
KNN tests on 1D toy data.
Creates M toy data instances (sample pairs) with eta samples per instance.
It's reasonable to let eta be small, so the data creation and training stage will go faster; 
the intention is just to measure regression speed as a function of T.

Each experiment considers K nearest neighbors and nonparametric estimator degree T.
An estimator is trained on the data; then regression is performed, both "full"
(using all the training data) and "KNN" (using only the K nearest neighbors). 

Note that only the speeds of the two regression strategies are measured. 
The appropriateness of the K value for this data must be determined separately 
via experiments with test error. 
"""
def KNN_tests_1D(M = 1000, eta = 1000, K = 5, Ts = range(1,21)):

    print ' >>> STARTING 1D KNN TESTS <<<'
    print

    print ' >>> Making', M, 'toy samples...'
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
        print ' >>> T value:', T
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

"""
KNN tests on D-dimensional simulation or simulation-style data.
Selects M data instances (sample pairs) with eta samples per instance. 
It's reasonable to let eta be small, so the data extraction and training stage will go faster; 
the intention is just to measure regression speed as a function of T.
Data need not be contiguous or otherwise meaningful, since this is just a test of speed.

Each experiment considers K nearest neighbors and nonparametric estimator degree T.
An estimator is trained on the data; then regression is performed, both "full"
(using all the training data) and "KNN" (using only the K nearest neighbors). 

Note that only the speeds of the two regression strategies are measured. 
The appropriateness of the K value for this data must be determined separately 
via experiments with test error. 
"""
def KNN_tests_ND(M = 5000, eta = 100, K = 10, Ts = range(1, 11), D = 3):

    print ' >>> STARTING KNN TESTS IN', D, 'DIMS <<<'

    print
    print ' >>> Extracting data with M =', M, ' eta =', eta, ' dim =', D
    data = manager.load_partial('sims/new_sim1_exact.txt', M, D, eta)

    print
    print ' >>> Length of data:', len(data)
    print ' >>> Number of samples per bin:',len(data[0])
    print ' >>> Dimensionality of each sample:',len(data[0][0])

    print 
    print ' >>> Scaling data across,', D, 'axes...'
    print ' >>> Before scaling,'
    print ' >>> >>> min max col 0:', manager.col_min_max(data, 0)
    for i in range(D):
        data = manager.scale_col_emp(data, i)
    print ' >>> After scaling,'
    print ' >>> >>> min max col 0:', manager.col_min_max(data, 0)

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

    for T in Ts:

        print
        print ' >>> T value:',T
        print ' >>> Training estimator... '
        start = time.clock()
        E = ND_Estimator(train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = T, dim = D)
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
            X0_coeffs = fourier_coeffs_ND(X0_sample, degree=T, dim=D)
            'number of test coeffs:',len(X0_coeffs)
            Y0_coeffs = E.KNN_regress(X0_coeffs, k=K)
        KNN_regress_times.append(time.clock() - start)
        print ' >>> KNN regression time:', KNN_regress_times[-1]

        print ' >>> Performing full regression...'
        start = time.clock()
        for i in range(len(test_samples)):            
            X0_sample, Y0_sample = test_samples_in[i], test_samples_out[i]
            X0_coeffs = fourier_coeffs_ND(X0_sample, degree=T, dim=D)
            Y0_coeffs = E.full_regress(X0_coeffs)
        full_regress_times.append(time.clock() - start)
        print ' >>> Full regression time:', full_regress_times[-1]

    print
    print ' >>> Experiments complete.'
    print ' >>> Ball build times:', ball_build_times
    print ' >>> KNN regress times:', KNN_regress_times
    print ' >>> Full regress times:', full_regress_times
    

"""
Measures how long it takes to fit N 6D particles to degree deg.
Makes plots in both standard and semilog scale.

Must import pylab in order to make plots.
"""
def coeff_tests_6D(N = 100, max_deg = 6, show_now = False):

    print ' >>> STARTING 6D COEFF TESTS <<<'

    mini_data = manager.load_floats('sims/sim1_approx_1000.txt')[:N]
    degrees = range(max_deg)
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

    if show_now: show()

"""
Divides specified data into num_bins divisions along each axis. 
Counts particles that lie in each of the resulting bins;
prints out resulting density distribution. 

For Hy's simulations (2^30 particles each), setting M = eta
gives bins_per_axis = int(32768**(1./3)). 
"""
def density_tests(div_per_axis = 5, infile = 'sims/new_sim1_exact.txt'):    

    print ' >>> STARTING DENSITY TESTS <<<'
    print ' >>> infile:',infile

    num_bins = div_per_axis**3
    print ' >>> divisions per dimension:', div_per_axis
    print ' >>> total num bins:', num_bins

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/div_per_axis
    binsz_y = (ymax - ymin)/div_per_axis
    binsz_z = (zmax - zmin)/div_per_axis

    print
    print ' >>> binsz_x:',binsz_x
    print ' >>> binsz_y:',binsz_y
    print ' >>> binsz_z:',binsz_z

    bindices = manager.bindices_3D(div_per_axis)
    print
    print ' >>> bindices:\t',bindices
    print

    manager.count_particles_3D(infile, bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, div_per_axis, verbose=True)

"""
Empirically computes Jhat of specified data, for specified dimension
(i.e. considering the first dim axes), for various values of 
the nonparametric estimator degree T.
"""
def Jhat_tests(infile = 'ex_bin_18.txt', dim = 3, Ts = range(10)):

    print ' >>> STARTING JHAT TESTS <<<'

    sample = np.loadtxt(infile)
    print
    print ' >>> infile:', infile
    print ' >>> len sample:', len(sample)

    samp = [sample]
    for col in range(3):
        samp = manager.scale_col_emp(samp, col)
    samp = samp[0]
    sample = np.column_stack((samp[:,0], samp[:,1], samp[:,2]))

    for T in Ts:
        print
        print ' >>> >>> degree (T):', T, '\tJhat:', J_hat_ND(sample, T, dim)

"""
Divides specified data into num_bins divisions along each axis. 
Finds all particles in specified bin (defaults to bin nearest origin).
Saves those particles to outfile.

For Hy's simulations (2^30 particles each), setting M = eta
gives bins_per_axis = int(32768**(1./3)). 
"""
def isolate_particles(div_per_axis = 18, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = None):    

    if (not outfile): outfile = 'innermost_bin_' + str(div_per_axis) + '.txt'

    print
    print ' >>> STARTING PARTICLE ISOLATION <<< \n'
    print ' >>> infile:',infile
    print ' >>> outfile:',outfile

    num_bins = div_per_axis**3
    print ' >>> divisions per dimension:', div_per_axis
    print ' >>> total num bins:', num_bins

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/div_per_axis
    binsz_y = (ymax - ymin)/div_per_axis
    binsz_z = (zmax - zmin)/div_per_axis

    print
    print ' >>> binsz_x:',binsz_x
    print ' >>> binsz_y:',binsz_y
    print ' >>> binsz_z:',binsz_z

    ps = manager.load_bin_3D(infile, bindex, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, verbose=True)
    print '\n >>> writing bin... \n'
    my_writetxt(outfile, ps)


"""
Preliminary demonstration of how to learn the identity distribution on simulation data. 
"""
def ID_demo():

    num_bins = 18
    bindex = [0, 0, 0]

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/num_bins
    binsz_y = (ymax - ymin)/num_bins
    binsz_z = (zmax - zmin)/num_bins

    # rescale 

    new_num_bins = 50

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

    input_ps = []
    assignments = manager.assign_particles_3D('ex_bin_18.txt', new_bindices, new_xmin, new_ymin, new_zmin, new_binsz_x, new_binsz_y, new_binsz_z, new_num_bins, verbose=True, chunk=10000)
    for key in sorted(assignments.keys()):
        cur = assignments[key]
        if len(cur) >= 5: input_ps.append(cur)

    print
    print 'number of training instances available:', len(input_ps)
    print 'avg. points/instance:', np.average([len(p) for p in input_ps])

    # run algorithm with some fraction of inputs

    for i in range(3):
        input_ps = manager.scale_col_emp(input_ps, i)

    # if desired: choose some subset of the training samples here. 
    # run following tests for different sizes

    dim = 3
    T = 3
    partial_lengths = [10, 50, 100, 150, 200, 250]
    errs = []

    for length in partial_lengths:
        
        partial_input = input_ps[:length]
        [train_samples, cv_samples, test_samples] = manager.partition_data(partial_input)

        train_samples_in = train_samples
        train_samples_out = train_samples
        cv_samples_in = cv_samples
        cv_samples_out = cv_samples
        test_samples_in = test_samples
        test_samples_out = test_samples

        print 
        print 'num training samples:', len(train_samples)
        print 'num cv / test samples:', len(cv_samples)
        
        E = ND_Estimator(train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = T, dim = dim)
        E.train(parallel=False)

        # compute average test error on test samples
        avg_err = np.average(test_errs_3D(E, test_samples_in, test_samples_out))
        print
        print 'average test error:', avg_err
        errs.append(avg_err)

    print
    for i in range(len(partial_lengths)):
        print 'data length:', partial_lengths[i],'avg. test error:',errs[i]

"""
Preliminary demonstration of how to perform regression on simulation data.
"""
def regression_demo():

    num_bins = 18
    bindex = [0, 0, 0]

    (xmin, xmax) = constants.col_0_min_max
    (ymin, ymax) = constants.col_1_min_max
    (zmin, zmax) = constants.col_2_min_max

    binsz_x = (xmax - xmin)/num_bins
    binsz_y = (ymax - ymin)/num_bins
    binsz_z = (zmax - zmin)/num_bins

    new_num_bins = 50

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

    inp_assignments = manager.assign_particles_3D('ex_bin_18.txt', new_bindices, new_xmin, new_ymin, new_zmin, new_binsz_x, new_binsz_y, new_binsz_z, new_num_bins, verbose=True, chunk=10000)
    outp_assignments = manager.assign_particles_3D('ex_bin_18_approx.txt', new_bindices, new_xmin, new_ymin, new_zmin, new_binsz_x, new_binsz_y, new_binsz_z, new_num_bins, verbose=True, chunk=10000)
    input_ps, output_ps = [], []

    for key in sorted(inp_assignments.keys()):
        cur_inp = inp_assignments[key]
        cur_outp = outp_assignments[key]
        if (len(cur_inp) >= 5 and len(cur_outp) >= 5): 
            input_ps.append(cur_inp)
            output_ps.append(cur_outp)

    print
    print 'number of training instances available:', len(input_ps)
    print 'avg. points/instance:', np.average([len(p) for p in input_ps])

    for i in range(3):
        input_ps = manager.scale_col_emp(input_ps, i)
        output_ps = manager.scale_col_emp(output_ps, i)

    dim = 3
    T = 5
    partial_lengths = [10, 50, 100, 150, 200, 250, 300, 350]
    errs = []

    for length in partial_lengths:
        
        partial_input = input_ps[:length]
        partial_output = output_ps[:length]
        [train_samples_in, cv_samples_in, test_samples_in] = manager.partition_data(partial_input)
        [train_samples_out, cv_samples_out, test_samples_out] = manager.partition_data(partial_output)

        E = ND_Estimator(train_samples_in, train_samples_out, cv_samples_in, cv_samples_out, test_samples_in, test_samples_out, degree = T, dim = dim)
        E.train(parallel=False)

        # compute average test error on test samples
        avg_err = np.average(test_errs_3D(E, test_samples_in, test_samples_out))
        print
        print 'average test error:', avg_err
        errs.append(avg_err)

    print
    for i in range(len(partial_lengths)):
        print 'data length:', partial_lengths[i],'avg. test error:',errs[i]


def tests():
    #KNN_tests_1D()
    #KNN_tests_ND()
    #coeff_tests_6D()
    density_tests(infile = 'sims/new_sim1_approx.txt')
    #Jhat_tests()
    pass

def demo():

    #isolate_particles()
    #ID_demo()
    #regression_demo()
    pass

"""
Runs any tests installed in tests(); runs demo().
"""
if __name__ == '__main__':

    print '\n > RUNNING TESTS \n'
    tests()

    print '\n > RUNNING DEMO \n'
    demo()
