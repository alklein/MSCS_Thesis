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
from math_helpers import *

"""
Returns "raw" values in file called filename (values left as strings)
"""
def raw_load(filename):
    return np.array([line.split() for line in open(filename)])

"""
Returns values in file called filename as array of arrays of floats
"""
def load_floats(filename):
    result = []
    for line in open(filename):
        result.append([float(val) for val in line.split()])
    return np.array(result)

"""
Parses and prints first how_many lines of file called filename
"""
def peek_floats(filename, how_many):
    print '\nPeeking at',filename,'\n'
    count = 1
    for line in open(filename):
        print([float(val) for val in line.split()])
        if (count >= how_many): return
        else: count += 1
    return 

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

"""
Extracts all particles in the specified bin from the file.

Assumes a 3D partitioning on the data; i.e. bindex is a 3D
vector of the form [i, j, k] that specifies the bin.
"""
def load_bin_3D(filename, bindex, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, verbose=False):
    [i, j, k] = bindex
    inner_x, outer_x = xmin + i*binsz_x, xmin + (i + 1)*binsz_x
    inner_y, outer_y = ymin + j*binsz_y, ymin + (j + 1)*binsz_y
    inner_z, outer_z = zmin + k*binsz_z, zmin + (k + 1)*binsz_z

    if (verbose):
        print
        print 'inner x:',inner_x,'outer x:',outer_x
        print 'inner y:',inner_y,'outer y:',outer_y
        print 'inner z:',inner_z,'outer z:',outer_z
        print

    ps = []
    count = 0
    for line in open(filename):

        if ((count % 10000000 == 0) and (verbose)): 
            print count/1000000,'million particles searched...'
            print 'particles found so far:',len(ps)
        count += 1

        cur_p = [float(val) for val in line.split()]
        x, y, z = cur_p[0], cur_p[1], cur_p[2]
        if ((inner_x < x) and (x <= outer_x) and (inner_y < y) and (y <= outer_y) and (inner_z < z) and (z <= outer_z)): 
            ps.append(cur_p)
    
    return np.array(ps)

"""
List of all bindices for 3D data partitioned 
into div_per_axis divisions along each axis
(for a total of 3^{div_per_axis} bins)
"""
def bindices_3D(div_per_axis):
    return [[i, j, k] for i in range(div_per_axis)
            for j in range(div_per_axis)
            for k in range(div_per_axis)]

"""
Maps bindices to their particle counts in 3D.
"""
def count_particles_3D(filename, bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=False):

    counts = {str(bindex) : 0 for bindex in bindices}

    xcuts = [xmin + i*binsz_x for i in range(num_bins)]
    ycuts = [ymin + i*binsz_y for i in range(num_bins)]
    zcuts = [zmin + i*binsz_z for i in range(num_bins)]
    cuts = [xcuts, ycuts, zcuts]

    count = 0
    for line in open(filename):

        if ((count % 10000000 == 0) and (verbose)):
            print ' >>>',count/1000000,'million particles searched'
        count += 1

        cur_p = [float(val) for val in line.split()]
        cur_bindex = []
        for i in range(3):
            
            cur_cuts = cuts[i]
            val = cur_p[i]

            index = 0
            next_cut = cur_cuts[index + 1]
          
            if (val >= cur_cuts[-1]): 
                cur_bindex.append(len(cur_cuts) - 1)
            else:
                while ((index < len(cur_cuts) - 2) and (next_cut < val)):
                    index += 1
                    next_cut = cur_cuts[index + 1]
                cur_bindex.append(index)
        if (str(cur_bindex) in counts):
            counts[str(cur_bindex)] += 1

    print ' >>> final counts:'
    for key in counts:
        if counts[key] > 0:
            print key,'-',counts[key]

    all_counts = [counts[key] for key in counts]
    print
    print all_counts

"""
Maps bindices to their particles in 3D.
"""
def assign_particles_3D(filename, bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=False, chunk=1000000):

    assignments = {str(bindex) : [] for bindex in bindices}

    xcuts = [xmin + i*binsz_x for i in range(num_bins)]
    ycuts = [ymin + i*binsz_y for i in range(num_bins)]
    zcuts = [zmin + i*binsz_z for i in range(num_bins)]
    cuts = [xcuts, ycuts, zcuts]

    count = 0
    for line in open(filename):

        if ((count % chunk == 0) and (verbose)):
            print 
            print count/chunk,'million particles searched'
            print 'current assignments:'
            for key in assignments:
                pass
        count += 1

        cur_p = [float(val) for val in line.split()]
        cur_bindex = []
        for i in range(3):
            
            cur_cuts = cuts[i]
            val = cur_p[i]

            index = 0
            next_cut = cur_cuts[index + 1]
          
            if (val >= cur_cuts[-1]): 
                cur_bindex.append(len(cur_cuts) - 1)
            else:
                while ((index < len(cur_cuts) - 2) and (next_cut < val)):
                    index += 1
                    next_cut = cur_cuts[index + 1]
                cur_bindex.append(index)
        if (str(cur_bindex) in assignments):
            assignments[str(cur_bindex)].append(cur_p)

    return assignments                

"""
Saves assignment dictionary to file.
"""
def save_assignments_3D(assignments, T, filename, xmin, xmax, ymin, ymax, zmin, zmax):
    for bindex in assignments:
        print
        ps = assignments[bindex]
        cut_ps = [p[:3] for p in ps]
        scaled_ps = [[(p[0] - xmin) / (xmax - xmin), (p[1] - ymin) / (ymax - ymin), (p[2] - zmin) / (zmax - zmin)] for p in cut_ps]
        coeffs = fourier_coeffs_ND(scaled_ps, T, 3)
        outp = ''
        for c in coeffs: outp += str(c) + ' '
        outp += '\n'
        print outp
        f = open(filename, 'a')
        f.write(outp)
        f.close()


"""
Returns empirical min and max values of an entire dataset
along some axis (the specified column). Stores axis in memory.
"""
def global_min_max(filename, col, verbose=False):
    vals = []
    count = 0
    for line in open(filename):
        cur_p = [float(val) for val in line.split()]
        vals.append(cur_p[col])
        if (count % 1000000 == 0): print count/1000000
        count += 1
    return (min(vals), max(vals))

"""
Returns empirical min and max values of an entire dataset
along some axis (the specified column). Uses constant memory.
"""
def lowmem_global_min_max(filename, col, verbose=False, start_min = 1000000, start_max = -1000000):
    cur_min, cur_max = start_min, start_max
    count = 0
    for line in open(filename):
        cur_p = [float(val) for val in line.split()]
        cur_val = cur_p[col]
        if (cur_val < cur_min): cur_min = cur_val
        if (cur_val > cur_max): cur_max = cur_val
        if (count % 1000000 == 0): 
            print count/1000000,'- cur min:',cur_min,'- cur max:',cur_max
        count += 1
    return (cur_min, cur_max)

"""
Returns min and max position and velocity values.

Note: data should be list of 6D samples,
where each sample is of the form [x, y, z, vx, vy, vz]
"""
def emp_bounds_6D(data): 
    xs, ys, zs = data[:,0], data[:,1], data[:,2]
    vxs, vys, vzs = data[:,3], data[:,4], data[:,5]
    min_pos = min([min(xs), min(ys), min(zs)])
    max_pos = max([max(xs), max(ys), max(zs)])
    min_vel = min([min(vxs), min(vys), min(vzs)])
    max_vel = max([max(vxs), max(vys), max(vzs)])
    return [min_pos, max_pos, min_vel, max_vel]

"""
Returns (min, max) values in column col of data.
"""
def col_min_max(data, col):
    data_col = []
    for sample in data:
        for row in sample:
            val = row[col]
            data_col.append(val)

    mn, mx = min(data_col), max(data_col)
    return (mn, mx)

"""
"Empirically" scales data along axis col 
by measuring the min and max values in that
column and then re-scaling to a (0, 1) range.
"""
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

"""
Partitions data according to specified holdout fraction.
"""
def partition_data(data, holdout_frac=.1):
    holdout_sz = int(holdout_frac * len(data))
    cv_samples = data[:holdout_sz]
    test_samples = data[holdout_sz : 2*holdout_sz]
    train_samples = data[2*holdout_sz : ]
    return [train_samples, cv_samples, test_samples]

"""
Returns number of lines in file called filename.
"""
def length(filename):
    i = 0
    for line in open(filename):
        i += 1
    return i

"""
Saves first count lines of file called infile as outfile.
"""
def make_mini(infile, outfile, count):
    i = 0
    data = []
    for line in open(infile):
        if i >= count: break
        data.append([float(val) for val in line.split()])
        i += 1
    np.savetxt(outfile, np.array(data))

def make_mini_files(parent_sim_exact, parent_sim_approx):
    print '\nMAKING MINI SIM FILES'

    print '\n --- sim1_exact_1000'
    make_mini(parent_sim_exact, 'sim1_exact_1000.txt', 1000)
    print ' --- > length:',length('sim1_exact_1000.txt') 
    
    print '\n --- sim1_exact_10000'
    make_mini(parent_sim_exact, 'sim1_exact_10000.txt', 10000)
    print ' --- > length:',length('sim1_exact_10000.txt') 
    
    print '\n --- sim1_exact_100000'
    make_mini(parent_sim_exact, 'sim1_exact_100000.txt', 100000)
    print ' --- > length:',length('sim1_exact_100000.txt') 
    
    print '\n --- sim1_exact_1000000'
    make_mini(parent_sim_exact, 'sim1_exact_1000000.txt', 1000000)
    print ' --- > length:',length('sim1_exact_1000000.txt') 
    
    print '\n --- sim1_exact_10000000'
    make_mini(parent_sim_exact, 'sim1_exact_10000000.txt', 10000000)
    print ' --- > length:',length('sim1_exact_10000000.txt') 

    print '\n --------------- '
    
    print '\n --- sim1_approx_1000'
    make_mini(parent_sim_approx, 'sim1_approx_1000.txt', 1000)
    print ' --- > length:',length('sim1_approx_1000.txt') 
    
    print '\n --- sim1_approx_10000'
    make_mini(parent_sim_approx, 'sim1_approx_10000.txt', 10000)
    print ' --- > length:',length('sim1_approx_10000.txt') 
    
    print '\n --- sim1_approx_100000'
    make_mini(parent_sim_approx, 'sim1_approx_100000.txt', 100000)
    print ' --- > length:',length('sim1_approx_100000.txt') 

    print '\n --- sim1_approx_1000000'
    make_mini(parent_sim_approx, 'sim1_approx_1000000.txt', 1000000)
    print ' --- > length:',length('sim1_approx_1000000.txt') 

    print '\n --- sim1_approx_10000000'
    make_mini(parent_sim_approx, 'sim1_approx_10000000.txt', 10000000)
    print ' --- > length:',length('sim1_approx_10000000.txt') 


if __name__ == '__main__':
    make_mini_files(parent_sim_exact = 'sims/new_sim1_exact.txt', parent_sim_approx = 'sims/new_sim1_approx.txt')

