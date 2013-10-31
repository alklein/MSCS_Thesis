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

import time
import math
import numpy as np
import scipy.integrate

from random import *
from pylab import *

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
f1, f2 are nonparametric estimator functions generated via approx_density().
"""
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

def triangle_kernel(x):
    if (0 <= x <= 1):
        return 1. - abs(x)
    else:
        return 0.

def RBF_kernel(x):
    return math.exp(-(x**2)/2.)

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
    def __init__(self, training_sample, cv_sample, num_terms = 20, dist_fn = L1_distance, kernel = triangle_kernel, bandwidths = [.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.]):

        self.max_weighted_output = None 
        self.num_terms = num_terms
        self.dist_fn = dist_fn
        self.kernel = kernel        
        self.bandwidths = bandwidths
        self.best_b = None

        # L1 norm -> function representation of nonparametric estimators
        if (dist_fn == L1_distance):
            self.nonparametric_estimation = approx_density
            self.L2_mode = False
        # L2 norm -> coefficient vector representation of nonparametric estimators
        elif (dist_fn == L2_distance):
            self.nonparametric_estimation = fourier_coeffs
            self.L2_mode = True

        self.Xs = training_sample[:,0]
        self.Ys = training_sample[:,1]
        self.cv_Xs = cv_sample[:,0]
        self.cv_Ys = cv_sample[:,1]

    def train(self):

        # fit training and cv data via nonparametric density estimation
        print ' >>> [debug] fitting training data... '
        self.X_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.Xs]
        self.Y_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.Ys]
        print ' >>> [debug] fitting cv data... '
        self.cv_X_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.cv_Xs]
        self.cv_Y_hats = [self.nonparametric_estimation(sample, self.num_terms) for sample in self.cv_Ys]

        # cross-validate bandwidths
        print ' >>> [debug] cross-validating bandwidths...'
        b_errs = []
        for b in self.bandwidths:
            net_err = 0.
            for i in range(len(self.cv_Xs)):
                input_coeffs = self.cv_X_hats[i]
                target_coeffs = self.cv_Y_hats[i]
                (Y0_fn, Y0_coeffs) = self.regress(input_coeffs, b=b)
                net_err += L2_distance(target_coeffs, Y0_coeffs)

            avg_err = net_err / (1.*len(self.cv_Xs))
            print ' >>> >>> [debug] average L2 error for bandwidth',b,'-',avg_err 
            b_errs.append(avg_err)
        print 'best b:', self.bandwidths[np.argmin(b_errs)]
        self.best_b = self.bandwidths[np.argmin(b_errs)]

    """
    given coeffs fit to some input sample_0, estimates the expected output distribution.
    returns estimator in function form and, if in L2 mode, coefficient form as well.
    (in L1 mode, the coefficient form is None.)
    """
    def regress(self, f0, b = None):

        if (not b): b = self.best_b # possibly still None

        distances = np.array([self.dist_fn(f0, f) for f in self.X_hats])
        normed_distances = distances / b

        """
        start = time.clock()
        distances = []
        for f in self.X_hats:
            d = self.dist_fn(f0, f)
            distances.append(d)
        elapsed = time.clock() - start
        print ' >>> [timer] TOTAL TIME TO COMPUTE PAIRWISE DISTANCES:',elapsed
        print ' >>> [timer] AVG TIME PER PAIR:',elapsed/len(self.X_hats)
        distances = np.array(distances)
        """

        #print ' >>> [debug] computing weights'
        k_sum = sum([self.kernel(d) for d in normed_distances])        
        #print ' >>> [debug] kernel sum:',k_sum
        weights = [self.kernel(normed_distances[i]) / k_sum for i in range(len(self.X_hats))]
        #print ' >>> [debug] sum of weights:',sum(weights)

        def Y0_fn(x):
            return sum([self.Y_hats[i](x) * weights[i] for i in range(len(self.Y_hats))])

        Y0_coeffs = None
        if (self.L2_mode):
            Y0_coeffs = np.zeros(self.num_terms)
            for i in range(len(self.Y_hats)):
                row = self.Y_hats[i]
                weight = weights[i]
                weighted_row = np.array([weight*val for val in row])
                Y0_coeffs += weighted_row
        else:
            self.max_weighted_output = self.Y_hats[np.argmax(weights)] # temp

        return (Y0_fn, Y0_coeffs)

    
class toyData:

    """
    Makes new toyData object.
    """
    def __init__(self, M=100, eta=100, holdout_frac=.1, verbose=True):

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


    M, eta = 100, 100

    print
    print ' > RUNNING BUILT-IN TESTS'

    """
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
    """

    print
    print ' > [debug] Making new toyData object...'
    tD = toyData(M = M, eta = eta)
    print ' > [debug] Checking param values...'
    tD.print_params()
    print ' > [debug] Generating toy training data...'

    tD.make_samples()
    all_data = tD.all_samples
    train_data = tD.train_samples
    cv_data = tD.cv_samples
    test_data = tD.test_samples

    print ' > [debug] Total number of toy data instances:', len(all_data)
    print ' > [debug] Number of training instances:', len(train_data)
    print ' > [debug] Number of cv instances:', len(cv_data)
    print ' > [debug] Number of test instances:', len(test_data)
    print 
    print ' > [debug] Length of input, output pairs:', len(train_data[0])
    print ' > [debug] Number of samples per distribution:', len(train_data[0][0])

    X0_sample, Y0_sample = test_data[0][0], test_data[0][1]
    print
    print ' > [debug] Creating estimator... '
    E = Estimator(train_data, cv_data, dist_fn = L2_distance, kernel = RBF_kernel)
    print ' > [debug] Training estimator... '
    E.train()
    print
    print ' > [debug] Regressing on new sample... '
    print
    X0_coeffs = fourier_coeffs(X0_sample, 20)
    (Y0_fn, Y0_coeffs) = E.regress(X0_coeffs)
    Y0_hat = coeffs_to_approx_density(Y0_coeffs)

    Y0 = coeffs_to_approx_density(fourier_coeffs(Y0_sample, 20))
    print
    print ' > [debug] Making plots... '
    print

    xs = np.array(range(100))/100.

    figure(1000)
    hist(Y0_sample, bins=100, normed=True, color='r')
    plot(xs, map(Y0, xs), linewidth=2, color='b')
    plot(xs, map(Y0_hat, xs), linewidth=2, color='k')
    title('M: ' + str(M) + ' eta: ' + str(eta))
    axes = gca()
    axes.set_xlim(0, 1)
    axes.set_ylim(-1, 6)

    show()

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

    show()
    """
