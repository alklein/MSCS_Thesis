#!/usr/bin/python2.7

"""
@file toy.py
@brief tools to make, sample, and regress on toy 1-D distributions
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

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

"""
convenience function to plot distributions.
"""
def make_fig(dist_class, dist=None, mu = 0., sig = 1., xmin = -5., xmax = 5., fig_num=0, xlab=None, ylab=None, tit=None, tit_fontsz=30):
    if not dist: dist = dist_class(mu, sig)
    Xs = np.linspace(xmin, xmax, 100)
    Ys = [dist.eval(X) for X in Xs]
    figure(fig_num)
    plot(Xs, Ys, '-')
    if xlab: xlabel(xlab, fontsize=24)
    if ylab: ylabel(ylab, fontsize=24)
    if tit: title(tit, fontsize=tit_fontsz)

"""
draws count samples from pdf in range (xmin, xmax).
"""
def rejection_sample(xmin, xmax, pdf, count):
    results = []
    bns = np.linspace(xmin, xmax, 1000)
    fn_max = max([pdf(bns[i]) for i in range(len(bns))])
    while (len(results) < count):
        x = uniform(xmin, xmax)
        h = uniform(0, fn_max)
        if (h < pdf(x)):
            results.append(x)
    return results

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
list representation of fourier coefficients corresponding to nonparametric 
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
actual estimated density function corresponding to nonparametric
estimation of the distribution from which sample was drawn, 
computed directly from sample.

used to compute L1 distance, make plots.
"""
def approx_density(sample, num_terms):

    coeffs = fourier_coeffs(sample, num_terms)

    def f_hat(x):
        return sum([coeffs[index]*cosine_basis(index)(x) for index in range(num_terms)])

    return f_hat

"""
actual estimated density function corresponding to nonparametric
estimation of the distribution from which sample was drawn, 
computed from fourier coefficients.

used to make plots.
"""
def coeffs_to_approx_density(coeffs):

    def f_hat(x):
        return sum([coeffs[index]*cosine_basis(index)(x) for index in range(len(coeffs))])

    return f_hat

"""
Unbiased risk associated with fitting sample 
to the estimator described by coeffs.
"""
def J_hat(sample, coeffs):
    T = len(coeffs)
    n = len(sample)
    result = 0
    for j in range(T): 
        term = 0
        for i in range(n):
            term += (2./n) * ((cosine_basis(j)(sample[i]))**2 - (n + 1)*(coeffs[j])**2)
        result += term
    return (1./(n - 1))*result

"""
f1, f2 are nonparametric estimator functions generated via approx_density().
"""
# NOT CURRENTLY IN USE. TO USE, NEED TO ADD import scipy.integrate
def L1_distance(f1, f2):

    def func(x):
        return abs(f1(x) - f2(x))

    y, err = scipy.integrate.quad(func, 0., 1., limit=200)
    return y

"""
c1, c2 are coefficient vectors. they should have the same length.
"""
def L2_distance(c1, c2):
    return sum([(c1[i] - c2[i])**2 for i in range(len(c1))])

"""
i.e. decreasing line;
defined for x in [0, 1]
"""
def triangle_kernel(x):
    if (0 <= x <= 1):
        return 1. - abs(x)
    else:
        return 0.

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
def coeff_Map(X):
    return [fourier_coeffs(x, 20) for x in X]

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
Computes L2 errors on test instances
"""
def test_errs(E, test_data, parallel=False, num_processes=5, KNN=False, k=1):
    test_Xs, test_Ys = test_data[:,0], test_data[:,1]    
    if (not parallel):
        print ' >>> [debug] Computing L2 test errors sequentially... '
        if (KNN): print ' >>> [debug] < using K =',k,'>'
        test_X_hats = [E.nonparametric_estimation(sample, E.num_terms) for sample in test_Xs]
        test_Y_hats = [E.nonparametric_estimation(sample, E.num_terms) for sample in test_Ys]
        errs = []
        for i in range(len(test_X_hats)):
            if (KNN): est_Y_hat = E.KNN_regress(test_X_hats[i], k=k)
            else: est_Y_hat = E.regress(test_X_hats[i])
            err = E.dist_fn(test_Y_hats[i], est_Y_hat)
            errs.append(err)
    else:
        P = Pool(processes=num_processes,)
        print ' >>> [debug] Computing L2 test errors in parallel with',num_processes,'processes... '
        if (KNN): print ' >>> [debug] < using K =',k,'>'
        test_X_errs = P.map(err_Map, multi_partitioned(E, test_Xs, test_Ys, KNN, k, num_processes))
        errs = list(itertools.chain(*test_X_errs))
    return errs

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
        self.phi = norm_pdf_dist(0., 1.)
        PHI = norm_cdf_dist(0., 1.)
        self.denom = PHI.eval((1. - self.mu)/self.sig) - PHI.eval(-self.mu/self.sig)

    def eval(self, x):
        return (1./self.sig) * self.phi.eval((x - self.mu)/self.sig) / self.denom

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


class Estimator:

    """
    training_sample is a list of training instances;
    cv_sample is a list of "holdout" instances for cross-validation;
    each instance is a tuple of the form [in, out]_i.
    """
    def __init__(self, training_sample, cv_sample, num_terms = 20, dist_fn = L2_distance, kernel = triangle_kernel, nonparametric_estimation = fourier_coeffs, bandwidths = [.15, .25]): # bandwidths = [.15, .25, .5, .75, 1., 1.25, 1.5]): TEMP!!!! # NOTE: bandwidths should be tried in increasing order.

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

    def train(self, parallel=False, num_processes=5, verbose=False):

        start = time.clock()

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
            print ' >>> [debug] Fitting training data in parallel with',num_processes,'process(es)... '
            coeffs = P.map(coeff_Map, partitioned(self.Xs, num_processes))
            self.X_hats = list(itertools.chain(*coeffs))
            coeffs = P.map(coeff_Map, partitioned(self.Ys, num_processes))
            self.Y_hats = list(itertools.chain(*coeffs))
            print ' >>> [debug] Fitting cv data in parallel with',num_processes,'processes... '
            coeffs = P.map(coeff_Map, partitioned(self.cv_Xs, num_processes))
            self.cv_X_hats = list(itertools.chain(*coeffs))
            coeffs = P.map(coeff_Map, partitioned(self.cv_Ys, num_processes))
            self.cv_Y_hats = list(itertools.chain(*coeffs))

        if verbose: 
            if parallel: 
                print ' >>> [debug] Time to fit coefficients in parallel with',num_processes,'process(es):',time.clock() - start
            else:
                print ' >>> [debug] Time to fit coefficients sequentially:',time.clock()

        if verbose:
            print 'length of fit training data:',len(self.X_hats),'cv data:',len(self.cv_X_hats)

        # TEMP AS FUCK
        print 'TEMP: EXITING COMPUTATION'
        exit(0)

        # cross-validate bandwidths
        print ' >>> [debug] cross-validating bandwidths...'
        b_errs = []
        for b in self.bandwidths:
            net_err = 0.
            for i in range(len(self.cv_Xs)):
                input_coeffs = self.cv_X_hats[i]
                target_coeffs = self.cv_Y_hats[i]
                Y0_coeffs = self.regress(input_coeffs, b=b)
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
    given coeffs fit to some input sample_0, estimates the expected output distribution.
    returns estimator in function form and, if in L2 mode, coefficient form as well.
    (in L1 mode, the coefficient form is None.)
    """
    def regress(self, f0, b = None):

        if (not b): b = self.best_b # possibly still None

        normed_distances = np.array([self.dist_fn(f0, f) for f in self.X_hats]) / b
        k_sum = sum([self.kernel(d) for d in normed_distances])        
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(len(self.X_hats))]

        a = np.matrix.transpose(np.array(self.Y_hats))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs

    """
    like regress(), but only considers K nearest neighbors to f0
    from the training data.
    """
    def KNN_regress(self, f0, b = None, k = 1):

        if (not b): b = self.best_b # possibly still None

        normed_distances = np.array([self.dist_fn(f0, f) for f in self.X_hats]) / b
        sorted_Xs = np.array(self.X_hats)[normed_distances.argsort()][:k]
        sorted_Ys = np.array(self.Y_hats)[normed_distances.argsort()][:k]
        normed_distances = sorted(normed_distances)[:k]
        k_sum = sum([self.kernel(d) for d in normed_distances])        
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(len(sorted_Xs))]

        a = np.matrix.transpose(np.array(sorted_Ys))
        b = np.array([[w] for w in weights])
        Y0_coeffs = np.dot(a, b)
            
        return Y0_coeffs

    
class toyData:

    """
    Makes new toyData object.
    """
    def __init__(self, M=100, eta=100, holdout_frac=.1, verbose=False):

        # if we're calling M the number of training + cv instances:
        self.num_toy_instances = int(1.1 * M) 

        self.M = M
        self.eta = eta
        self.holdout_frac = holdout_frac

        self.all_samples = None
        self.train_samples = None
        self.cv_samples = None
        self.test_samples = None

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

    Two "holdout" pairs of size holdout_frac * M are set aside.

    all_samples: complete toy data
    train_samples: used to "train" regressor
    cv_samples (holdout 1): used to cross-validate bandwidth
    test_samples (holdout 2): used to test cross-validated regressor
    """
    def make_samples(self):
        samples = []
        for i in range(self.num_toy_instances):

            # mu_1, mu_2 ~ Unif[0, 1]
            mu_1 = np.random.rand()
            mu_2 = np.random.rand()

            # sig_1, sig_2 ~ Unif[0.05, 0.10]
            sig_1 = .05*(np.random.rand() + 1)
            sig_2 = .05*(np.random.rand() + 1)

            # create and sample probability distributions
            p = p_dist(mu_1, mu_2, sig_1, sig_2)
            q = q_dist(mu_1, mu_2, sig_1, sig_2)
            input_samples = rejection_sample(0., 1., p.eval, self.eta)
            output_samples = rejection_sample(0., 1., q.eval, self.eta)
            samples.append([input_samples, output_samples])

        self.all_samples = np.array(samples)

        holdout_sz = int(self.holdout_frac * self.M)
        self.cv_samples = self.all_samples[:holdout_sz]
        self.test_samples = self.all_samples[holdout_sz:2*holdout_sz]
        self.train_samples = self.all_samples[2*holdout_sz:]


    """
    Dumps all of this object's samples to a file.
    Instances are separated by newlines.
    Input, Output pairs are separated by semicolons.
    Samples (from either an input or an output function) are separated by commas.

    When not in append mode, clobbers any existing file with this filename.
    """
    def save_samples(self, filename, append=False):
        if append:
            try:
                # open existing file in append mode
                f = open(filename, 'a+')
            except: 
                # if file does not exist, create it and write from start
                np.savetxt(filename, [])
                f = open(filename, 'a+')
        else:
            # clobber file if it exists; create new file and write from start
            os.system('rm ' + filename)
            np.savetxt(filename, [])
            f = open(filename, 'r+')
            
        for instance in self.all_samples:
            inp, outp = instance[0], instance[1]
            s1, s2 = '', ''
            for val in inp: s1 += str(val) + ','
            for val in outp: s2 += str(val) + ','
            s1, s2 = s1[:-1], s2[:-1] # remove final commas
            f.write(s1 + ';' + s2 + '\n')
        f.close()

    """
    Attempts to load samples from file according to specified values of M, eta.
    """
    def load_samples(self, filename):
        data = []
        for instance in open(filename):
            raw_inp, raw_outp = instance.split(';')
            inp = [float(val) for val in raw_inp.split(',')]
            outp = [float(val) for val in raw_inp.split(',')]
            data.append([inp, outp])

        self.all_samples = np.array([line[:self.eta] for line in data[:self.M]])

        holdout_sz = int(self.holdout_frac * self.M)
        self.cv_samples = self.all_samples[:holdout_sz]
        self.test_samples = self.all_samples[holdout_sz:2*holdout_sz]
        self.train_samples = self.all_samples[2*holdout_sz:]

    """
    Debugging method.
    """
    def print_params(self):
        print
        print ' >>> [debug] M:',self.M
        print ' >>> [debug] eta:',self.eta
        print


"""
Tests related to the optimal number of terms, T,
to retain in the nonparametric estimator.
"""
def T_test():
    xs = np.array(range(100))/100.    
    dist = p_dist(.3, .6, .05, .07)
    #cutoffs = {100: 10, 1000: 15, 10000: 20, 100000: 25}    
    cutoffs = {100: 10, 1000: 15}

    for num_samples in cutoffs:
        sample = rejection_sample(0, 1, dist.eval, num_samples)
        figure(num_samples)
        hist(sample, bins=100, normed=True, color='0.75')
        cut = cutoffs[num_samples]
        for T in range(cut):
            f_hat = approx_density(sample, num_terms=T)
            cut = cutoffs[num_samples]
            plot(xs, map(f_hat, xs), linewidth=1, color='b')
        for T in range(cut, 40):
            f_hat = approx_density(sample, num_terms=T)
            cut = cutoffs[num_samples]
            plot(xs, map(f_hat, xs), linewidth=1, color='r')
        T = cut
        f_hat = approx_density(sample, num_terms=T)
        cut = cutoffs[num_samples]
        plot(xs, map(f_hat, xs), linewidth=4, color='k', linestyle='--')

    # shows how J_hat scales with T for different sample sizes
    for num_samples in cutoffs:
        sample = rejection_sample(0, 1, dist.eval, num_samples)
        figure(2*num_samples)
        Js = []
        for T in range(40):
            cs = fourier_coeffs(sample, num_terms=T)
            Js.append(J_hat(sample, cs))
        plot(range(40), Js, '-')        
        xlabel('T', fontsize=24)
        ylabel('J', fontsize=24)
        axvline(x = cutoffs[num_samples])

    show()

"""
Tests brute-force toy data creation vs. loading data from files.
"""
def load_speed_test():
    brute_times = []
    load_times = []
    Ms = [100, 500, 1000, 1500, 2000, 5000] #, 10000]
    num_trials = 2

    brute = [3.45, 24.25, 69.9, 137.33, 220.8, 1146.7]
    load = [13.77, 13.92, 14.02, 13.57, 14.02, 14.85]

    figure(0)
    semilogy(Ms, brute)
    semilogy(Ms, load)
    xlabel('Number of samples', fontsize=20)
    ylabel('Time to make (black) vs. load (red) data (s)', fontsize=20)
    show()

    for M in Ms:
        print 
        print ' ........................'
        print 
        print ' > Making toyData obj with M, eta =',M
        eta = M
        tD = toyData(M = M, eta = eta)

        print
        print ' > Making samples from scratch'
        times = 0
        for i in range(num_trials):
            start = time.clock()
            tD.make_samples()
            diff = time.clock() - start
            print ' > ... time:',diff
            times += diff
        avg_time = times/num_trials
        print ' > ... avg. time:',avg_time
        print ' > ... num samples made:',len(tD.all_samples)
        brute_times.append(avg_time)

        tD2 = toyData(M = M, eta = eta)

        print
        print ' > Loading samples from file'
        times = 0
        for i in range(num_trials):
            start = time.clock()
            tD2.load_samples('data.txt')
            diff = time.clock() - start
            print ' > ... time:',diff
            times += diff
        avg_time = times/num_trials
        print ' > ... avg. time:',avg_time
        print ' > ... num samples made:',len(tD2.all_samples)
        load_times.append(avg_time)

    print 'brute:',brute_times
    print 'load:',load_times

"""
Tests time to fit coefficients sequentially vs. in parallel.
"""
def pll_test():

    M, eta = 500, 500

    print
    print ' > [debug] Making new toyData object with M, eta =',M,'...'
    tD = toyData(M = M, eta = eta)
    tD.make_samples()

    all_data = tD.all_samples
    train_data = tD.train_samples
    cv_data = tD.cv_samples
    test_data = tD.test_samples

    print
    print ' > [debug] Training estimator... '
    E = Estimator(train_data, cv_data, dist_fn = L2_distance, kernel = RBF_kernel)
    E.train(parallel=False, verbose=True)    
    for n in range(2, 10):
        print
        E.train(parallel=True, num_processes=n, verbose=True)

def test():    

    #T_test()
    # load_speed_test()
    pll_test()
    exit(0)

def demo(num_plots = 1):

    M, eta = 1000, 1000

    print
    print ' > [debug] Making new toyData object...'
    tD = toyData(M = M, eta = eta)

    #print
    #print ' > [debug] Generating toy training data...'
    #tD.make_samples()

    #print
    #print ' > [debug] Writing toy data to file'
    #tD.save_samples('data.txt', append=True) 

    print
    print ' > [debug] Reading toy data from file... '
    tD.load_samples('data.txt')

    all_data = tD.all_samples
    train_data = tD.train_samples
    cv_data = tD.cv_samples
    test_data = tD.test_samples

    print
    print ' > [debug] Total number of toy data instances:', len(all_data)
    print ' > [debug] Number of training instances:', len(train_data)
    print ' > [debug] Number of cv instances:', len(cv_data)
    print ' > [debug] Number of test instances:', len(test_data)

    print
    print ' > [debug] Creating estimator... '
    E = Estimator(train_data, cv_data, dist_fn = L2_distance, kernel = RBF_kernel)
    print ' > [debug] Training estimator... '
    E.train(parallel=True)

    for i in range(num_plots):

        X0_sample, Y0_sample = test_data[i][0], test_data[i][1]
        print
        print ' > [debug] Regressing on new sample... '
        X0_coeffs = fourier_coeffs(X0_sample, 20)
        #Y0_coeffs = E.regress(X0_coeffs)
        Y0_coeffs = E.KNN_regress(X0_coeffs, k=1)
        X0_hat = coeffs_to_approx_density(X0_coeffs)
        Y0_hat = coeffs_to_approx_density(Y0_coeffs)        
        Y0 = coeffs_to_approx_density(fourier_coeffs(Y0_sample, 20))

        print ' > [debug] Making plots... '        
        xs = np.array(range(100))/100.
        
        figure(2*i)
        hist(X0_sample, bins=100, normed=True, color='r')
        plot(xs, map(X0_hat, xs), linewidth=2, color='b')
        title('INPUT. M: ' + str(M) + ' eta: ' + str(eta))
        axes = gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 6)

        figure(2*i + 1)
        hist(Y0_sample, bins=100, normed=True, color='r')
        plot(xs, map(Y0, xs), linewidth=2, color='b')
        plot(xs, map(Y0_hat, xs), linewidth=2, color='k')
#        title('OUTPUT. M: ' + str(M) + ' eta: ' + str(eta))
        axes = gca()
        axes.set_xlim(0, 1)
        axes.set_ylim(-1, 6)
        
#    show()

    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    avg_errs = [np.average(test_errs(E, test_data, parallel=True, KNN=True, k=ks[i])) for i in range(len(ks))]
    avg_errs.append(np.average(test_errs(E, test_data, parallel=True, KNN=False)))

    print
    for i in range(len(ks)):
        print ' > [debug] Average test error, k =',ks[i],':', avg_errs[i]
    print ' > [debug] Average test error, k = all:', avg_errs[-1]

    figure(2)
    plot(ks, avg_errs[:-1])
    axhline(y = avg_errs[-1])
    xlabel('K', fontsize=24)
    ylabel('Avgerage L2 Error', fontsize=24)
 #   title('M, eta = ' + str(eta), fontsize=30)
    show()


"""
Runs built-in tests and a demo.
"""
if __name__ == '__main__':

    print
    print ' > RUNNING BUILT-IN TESTS'
    test()

    print
    print ' > RUNNING DEMO'
    demo()

    """
    figure(1001)
    plot(xs, map(Y0, xs), linewidth=2, color='b')
    plot(xs, map(E.similar_output, xs), linewidth=2, color='r')
    plot(xs, map(E.max_weighted_output, xs), linewidth=1, color='g')
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 6)
    
    dist = p_dist(.3, .6, .05, .07)
    sample = rejection_sample(0, 1, dist.eval, 5000)
    f_hat = approx_density(sample, num_terms=25)

    figure(101)
    hist(sample, bins=100, normed=True, color='r')
    plot(xs, map(dist.eval, xs), linewidth=2, color='b')
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 5)

    figure(102)
    hist(sample, bins=100, normed=True, color='r')
    plot(xs, map(f_hat, xs), linewidth=2, color='k')
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 5)

    print
    print ' > [debug] testing distribution distance integral...'
    print 'distance between dist and itself:', distance(dist.eval, dist.eval)
    dist2 = p_dist(.25, .6, .05, .07)
    print 'distance between dist and similar dist:', distance(dist.eval, dist2.eval)
    dist3 = p_dist(.1, .9, .03, .08)
    print 'distance between dist and less similar dist:', distance(dist.eval, dist3.eval)

    print
    print ' > MAKING DISTRIBUTION PLOTS'

    #make_fig(norm_pdf_dist, tit='Normal PDF, CDF. Mu=0, Sig=1')
    #make_fig(norm_cdf_dist)
    #make_fig(g_dist, fig_num=1, tit='G Distribution. Mu=0, Sig=1')
    make_fig(p_dist, dist=p_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2)
    #make_fig(p_dist, dist=p_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2, tit='P and Q. Mu1=.3, Mu2=.6, Sig1=.05, Sig2=.07', tit_fontsz=24)
    make_fig(q_dist, dist=q_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2)

    print
    print ' > [debug] testing cosine basis...'
    phi_0 = cosine_basis(0)
    phi_1 = cosine_basis(1)
    phi_2 = cosine_basis(2)
    phi_3 = cosine_basis(3)

    xs = np.array(range(100))/100.

    figure(100)
    plot(xs, map(phi_0, xs))
    plot(xs, map(phi_1, xs))
    plot(xs, map(phi_2, xs))
    plot(xs, map(phi_3, xs))

    show()
    """
