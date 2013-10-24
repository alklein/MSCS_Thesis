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
import scipy.integrate
import numpy as np

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

    y, err = scipy.integrate.quad(func, 0, 1)
    return y


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
    figure(100)
    plot(xs, map(phi_0, xs))
    plot(xs, map(phi_1, xs))
    plot(xs, map(phi_2, xs))
    plot(xs, map(phi_3, xs))

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


    #show()
