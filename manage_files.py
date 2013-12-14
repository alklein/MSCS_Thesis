#!/usr/bin/python2.7

"""
@file manage_files.py
@brief Tools to manage and manipulate simulation files
@author Andrea Klein     <alklein@alumni.stanford.edu>
"""

__author__ = "Andrea Klein"

import os
import sys
import time
import math
import itertools
import numpy as np

from random import *

def raw_load(filename):
    return np.array([line.split() for line in open(filename)])

def load_floats(filename):
    result = []
    for line in open(filename):
        result.append([float(val) for val in line.split()])
    return np.array(result)

"""
straight loading function (no sorting of particles; 
just loads in order and puts the right number in each bin.)
loads num bins of size binsz, for a total of num*binsz particles.
loads first dim attributes of each particle.
"""
def load_partial(filename, num_bins, dim, binsz):
    result = []
    cur_sample = []
    cur_num_bins = 0
    cur_particles_per_bin = 0

    for line in open(filename):
        if (cur_num_bins >= num_bins):
            # we're done. return everything
            return np.array(result)
        else:
            # put current particle into a bin
            row = [float(val) for val in line.split()]
            if (cur_particles_per_bin < binsz):
                # append to cur bin
                cur_sample.append(row[:dim]) 
                cur_particles_per_bin += 1
            else:
                # bin is done
                result.append(cur_sample)
                cur_num_bins += 1
                cur_sample = []
                cur_particles_per_bin = 0
    # if we run out of data, just return what we have:
    return np.array(result)

# data should be list of 6D samples,
# where each sample is of the form
# [x, y, z, vx, vy, vz]
def emp_bounds_6D(data): 
    xs, ys, zs = data[:,0], data[:,1], data[:,2]
    vxs, vys, vzs = data[:,3], data[:,4], data[:,5]
    min_pos = min([min(xs), min(ys), min(zs)])
    max_pos = max([max(xs), max(ys), max(zs)])
    min_vel = min([min(vxs), min(vys), min(vzs)])
    max_vel = max([max(vxs), max(vys), max(vzs)])
    return [min_pos, max_pos, min_vel, max_vel]


def col_min_max(data, col):
    data_col = []
    for sample in data:
        for row in sample:
            val = row[col]
            data_col.append(val)

    mn, mx = min(data_col), max(data_col)
    return (mn, mx)

def scale_col_emp(data, col):
    (mn, mx) = col_min_max(data, col)

    for i in range(len(data)):
        sample = data[i]
        for j in range(len(sample)):
            row = sample[j]
            val = row[col]
            scaled_val = (val - mn) / (mx - mn)
            data[i][j][col] = scaled_val

    return data

# TODO: implement
"""
def load_floats_scaled(filename, [xmin, xmax, vmin, vmax]):

    def sc_pos(x):
        return float(x) * (xmax - xmin) - xmin

    def sc_vel(v):
        return float(v) * (vmax - vmin) - vmin
    result = []
    for line in open(filename):
        [x, y, z, vx, vy, vz] = line.split()
        result.append([sc_pos(x), sc_pos(y), sc_pos(z), sc_vel(vx), sc_vel(vy), sc_vel(vz)])
    return np.array(result)
"""

"""
Partitions data according to specified holdout fraction.
"""
def partition_data(data, holdout_frac=.1):
    holdout_sz = int(holdout_frac * len(data))
    cv_samples = data[:holdout_sz]
    test_samples = data[holdout_sz : 2*holdout_sz]
    train_samples = data[2*holdout_sz : ]
    return [train_samples, cv_samples, test_samples]

def length(filename):
    i = 0
    for line in open(filename):
        i += 1
    return i

def make_mini(infile, outfile, count):
    i = 0
    data = []
    for line in open(infile):
        if i >= count: break
        data.append([float(val) for val in line.split()])
        i += 1
    np.savetxt(outfile, np.array(data))

def make_mini_files():
    print '\nMAKING MINI SIM FILES'

    print '\n --- sim1_exact_1000'
    make_mini('sim1_exact.txt', 'sim1_exact_1000.txt', 1000)
    print ' --- > length:',length('sim1_exact_1000.txt') 
    
    print '\n --- sim1_exact_10000'
    make_mini('sim1_exact.txt', 'sim1_exact_10000.txt', 10000)
    print ' --- > length:',length('sim1_exact_10000.txt') 
    
    print '\n --- sim1_exact_100000'
    make_mini('sim1_exact.txt', 'sim1_exact_100000.txt', 100000)
    print ' --- > length:',length('sim1_exact_100000.txt') 
    
    print '\n --- sim1_exact_1000000'
    make_mini('sim1_exact.txt', 'sim1_exact_1000000.txt', 1000000)
    print ' --- > length:',length('sim1_exact_1000000.txt') 
    
    print '\n --- sim1_exact_10000000'
    make_mini('sim1_exact.txt', 'sim1_exact_10000000.txt', 10000000)
    print ' --- > length:',length('sim1_exact_10000000.txt') 

    print '\n --------------- '
    
    print '\n --- sim1_approx_1000'
    make_mini('sim1_approx.txt', 'sim1_approx_1000.txt', 1000)
    print ' --- > length:',length('sim1_approx_1000.txt') 
    
    print '\n --- sim1_approx_10000'
    make_mini('sim1_approx.txt', 'sim1_approx_10000.txt', 10000)
    print ' --- > length:',length('sim1_approx_10000.txt') 
    
    print '\n --- sim1_approx_100000'
    make_mini('sim1_approx.txt', 'sim1_approx_100000.txt', 100000)
    print ' --- > length:',length('sim1_approx_100000.txt') 

    print '\n --- sim1_approx_1000000'
    make_mini('sim1_approx.txt', 'sim1_approx_1000000.txt', 1000000)
    print ' --- > length:',length('sim1_approx_1000000.txt') 

    print '\n --- sim1_approx_10000000'
    make_mini('sim1_approx.txt', 'sim1_approx_10000000.txt', 10000000)
    print ' --- > length:',length('sim1_approx_10000000.txt') 


if __name__ == '__main__':
    print 'length of sim1_exact.txt:', length('sims/sim1_exact.txt')
    #make_mini_files()

