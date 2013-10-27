#!/usr/bin/python

"""
@file toy.py
@brief tools to make, sample, and regress on toy 1-D distributions
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

import sys
# if necessary: specify location of scipy on next line, e.g.:
#sys.path.append('/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/py2app/recipes/')

import math
import numpy as np
import scipy.integrate

from random import *
from pylab import *

def make_fig(dist_class, dist=None, mu = 0., sig = 1., xmin = -5., xmax = 5., fig_num=0, xlab=None, ylab=None, tit=None, tit_fontsz=30):
    if not dist: dist = dist_class(mu, sig)
    Xs = np.linspace(xmin, xmax, 100)
    Ys = [dist.eval(X) for X in Xs]
    figure(fig_num)
    plot(Xs, Ys, '-')
    if xlab: xlabel(xlab, fontsize=24)
    if ylab: ylabel(ylab, fontsize=24)
    if tit: title(tit, fontsize=tit_fontsz)

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

# returns indexed function from the cosine basis
def cosine_basis(index):

    def one(x):
        return 1

    def phi(x):
        return (2**.5) * math.cos(math.pi * index * x)

    if (not index): return one
    else: return phi

def fourier_coeff(index, sample):
    phi = cosine_basis(index)
    return sum([phi(s) for s in sample])/(1.*len(sample))

def approx_density(sample, num_terms):
    coeffs = [fourier_coeff(index, sample) for index in range(num_terms)]

    def f_hat(x):
        return sum([coeffs[index]*cosine_basis(index)(x) for index in range(num_terms)])

    return f_hat

def distance(f1, f2):

    def func(x):
        return abs(f1(x) - f2(x))

    ### TEMP
    """
    figure(10)
    xs = np.array(range(100))/100.
    plot(xs, map(func, xs))
    title('function we are integrating')
    show()
    """
    ###
    y, err = scipy.integrate.quad(func, 0., 1., limit=200) # TEMP
    #print ' >>> [debug] ERROR upon integration:',err
    return y

def kernel(x):
    return 1 - abs(x)

# TODO: convert to list comprehension after checking
def kernel_sum(dist, all_dists, bandwidth):
    result = 0.
    for other_dist in all_dists:
        result += kernel(distance(dist, other_dist)/(1.*bandwidth))
    return result

def weight(dist_1, dist_2, bandwidth, k_sum):
    if k_sum > 0:
        return kernel(distance(dist_1, dist_2)/(1.*bandwidth))/(1.*k_sum)
    else:
        return 0.

# not quite right. should take training data of the form <P_i, Q_i>
"""
def estimator(dist_0, all_dists, num_terms):
    bandwidth = max([distance(dist_0, dist) for dist in all_dists])
    k_sum = kernel_sum(dist, all_dists, bandwidth)

    def q_0(x):
        result = 0.
        for i in range(num_terms):
            phi = cosine_basis(i)
            term = 0.
            for dist in all_dists:
                W = weight(dist_0, dist, bandwidth, k_sum)
                term += W * dist
            result += term * phi
        return result

    return q_0
"""

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
    training_sample is a length-M list of training instances;
    each instance is a tuple of the form [in, out]_i.

    during initialization, automatically computes:
    1. nonparametric approximations
    of the input and output distributions
    2. pairwise distances between the distributions
    """
    def __init__(self, training_sample, num_terms=20):

        self.similar_output = None # temp
        self.max_weighted_output = None # temp
        self.num_terms = 20
        Xs = training_sample[:,0]
        Ys = training_sample[:,1]
        self.X_hats = [approx_density(sample, num_terms) for sample in Xs]
        self.Y_hats = [approx_density(sample, num_terms) for sample in Ys]
        print
        print ' >>> [debug] number of training instances:',len(self.X_hats)

    """
    given a new input sample_0, estimates the expected output distribution
    """
    def regress(self, sample_0):

        print ' >>> [debug] approximating sample density fn for regression'
        f0 = approx_density(sample_0, self.num_terms)
        print ' >>> [debug] computing distance from f0 to each training dist'
        distances = np.array([distance(f0, f) for f in self.X_hats])
        #normed_distances = distances / (1.*sum(distances)) # normalize by bandwidth 
        # TEMP
        normed_distances = distances / (1. * max(distances)) # normalize by bandwidth 
        print ' >>> [debug] computing kernel sum'
        k_sum = sum([kernel(d) for d in normed_distances])        
        print ' >>> [debug] kernel sum:',k_sum

        print ' >>> [debug] computing weights'
        weights = [kernel(normed_distances[i]) / k_sum for i in range(len(self.X_hats))] # temp

        self.similar_output = self.Y_hats[np.argmin(normed_distances)] # temp
        self.max_weighted_output = self.Y_hats[np.argmax(weights)] # temp
        print ' >>> [debug] min-distance dist:', self.similar_output
        print ' >>> [debug] max-weight dist:', self.max_weighted_output

        """
        figure(0)
        hist(distances)
        title('distances')

        figure(1)
        hist(weights, bins=100)
        title('weights')
        """

        def Y0(x):
            return sum([self.Y_hats[i](x) * weights[i] for i in range(len(self.Y_hats))])
            
        return Y0
    
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
            input_samples = rejection_sample(0., 1., p.eval, self.eta)
            output_samples = rejection_sample(0., 1., q.eval, self.eta)
            samples.append([input_samples, output_samples])

        samples = np.array(samples)
        self.samples = samples
        return samples

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

    num_training_pairs = 2000
    num_testing_pairs = 1
    samples_per_dist = 2000


    print
    print ' > RUNNING BUILT-IN TESTS'

    print
    print ' > MAKING DISTRIBUTION PLOTS'

    """
    make_fig(norm_pdf_dist, tit='Normal PDF, CDF. Mu=0, Sig=1')
    make_fig(norm_cdf_dist)
    make_fig(g_dist, fig_num=1, tit='G Distribution. Mu=0, Sig=1')
    make_fig(p_dist, dist=p_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2)
    #make_fig(p_dist, dist=p_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2, tit='P and Q. Mu1=.3, Mu2=.6, Sig1=.05, Sig2=.07', tit_fontsz=24)
    make_fig(q_dist, dist=q_dist(.3, .6, .05, .07), xmin=0., xmax=1., fig_num=2)
    """

    print
    print ' > [debug] testing cosine basis...'
    phi_0 = cosine_basis(0)
    phi_1 = cosine_basis(1)
    phi_2 = cosine_basis(2)
    phi_3 = cosine_basis(3)
    xs = np.array(range(100))/100.

    """
    figure(100)
    plot(xs, map(phi_0, xs))
    plot(xs, map(phi_1, xs))
    plot(xs, map(phi_2, xs))
    plot(xs, map(phi_3, xs))
    """

    print
    print ' > [debug] Making new toyData object...'
    tD = toyData(M = num_training_pairs, eta = samples_per_dist)
    print ' > [debug] Checking param values...'
    tD.print_params()
    print ' > [debug] Generating toy training data...'
    train_data = tD.make_samples()
    print ' > [debug] Number of toy training instances:', len(train_data)
    print ' > [debug] Length of input, output pairs:', len(train_data[0])
    print ' > [debug] Number of samples per distribution:', len(train_data[0][0])
    print

    print ' > [debug] Making new toyData object...'
    tD2 = toyData(M = num_testing_pairs, eta = samples_per_dist)
    print ' > [debug] Generating toy testing data...'
    test_data = tD2.make_samples()
    print ' > [debug] Number of toy testing instances:', len(test_data)
    print ' > [debug] Length of input, output pairs:', len(test_data[0])
    print ' > [debug] Number of samples per distribution:', len(test_data[0][0])

    X0_sample, Y0_sample = test_data[0][0], test_data[0][1]
    E = Estimator(train_data)
    Y0_hat = E.regress(X0_sample)
    Y0 = approx_density(Y0_sample, 20)

    figure(1000)
    hist(Y0_sample, bins=100, normed=True, color='r')
    plot(xs, map(Y0, xs), linewidth=2, color='b')
    plot(xs, map(Y0_hat, xs), 'x', linewidth=2)
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 6)

    figure(1001)
    plot(xs, map(Y0, xs), linewidth=2, color='b')
    plot(xs, map(E.similar_output, xs), linewidth=2, color='r')
    plot(xs, map(E.max_weighted_output, xs), linewidth=1, color='g')
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 6)

    #figure(1001)
    #plot(xs, map(Y0_hat, xs), linewidth=2, color='k')

#    axes = gca()
#    axes.set_xlim(0, 1)
#    axes.set_ylim(-1, 5)
    
    show()

    """
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

    show()
    """
