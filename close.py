#!/usr/bin/python2.7

"""
@file close.py
@brief code used to examine simulation subsets
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

""" Custom Imports """
import toy
import constants

"""
Extracts all particles in the specified range from the file.

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

    print
    print ' >>> using exact column min max values... '
    (xmin, xmax) = constants.exact_col_0_min_max
    (ymin, ymax) = constants.exact_col_1_min_max
    (zmin, zmax) = constants.exact_col_2_min_max

    binsz_x = (xmax - xmin)/div_per_axis
    binsz_y = (ymax - ymin)/div_per_axis
    binsz_z = (zmax - zmin)/div_per_axis

    print
    print ' >>> binsz_x:',binsz_x
    print ' >>> binsz_y:',binsz_y
    print ' >>> binsz_z:',binsz_z

    ps = load_bin_3D(infile, bindex, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, verbose=True)
    print '\n >>> writing bin... \n'
    my_writetxt(outfile, ps)

# description: determines bins for use by vis function
# input:
     # data to 'visualize'
     # desired pixel density per axis
# output:
     # x and y values for use in plotting
     # bounds of plotting (to fit all data)
# effects: n/a
# notes: n/a

def bns_ext(data, res=400):
    xmins, xmaxs = map(min, data[0]), map(max, data[0])
    ymins, ymaxs = map(min, data[1]), map(max, data[1])
    xmin, xmax = min(xmins), max(xmaxs)
    ymin, ymax = min(ymins), max(ymaxs)
    xrange, yrange = np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res)
    return [yrange, xrange], [xmin, xmax, ymin, ymax]

# description: makes colored 'scatter' plots for side-by-side comparison of data
# input:
     # data = [[ x1, x2, ... ],[ y1, y2, ... ]]
     # labels = [ xlabel, ylabel ]
     # titles = [ title1, title2, ... ]
# output: n/a (makes graph)
# effects: n/a
# notes: does not take log of data (must use np.log beforehand for semilog or log-log vis)

def vis(data, labels, titles):
    i = 0
    while i < len(data[0]):
        figure(i)
        xs, ys = data[0][i], data[1][i]
        bns, ext = bns_ext(data)
        H, yedges, xedges = np.histogram2d(ys, xs, bins=bns)
        imshow(H, extent=ext, aspect='equal', cmap = cm.jet, interpolation='nearest', origin='lower', vmin=0.0, vmax=10.0)
        colorbar()
        rc('text', usetex=True)
        xlabel(labels[0], fontsize=20)
        ylabel(labels[1], fontsize=20)
        title(titles[i], fontsize=24)
        i += 1
    show()

def scan_min_max(filename):
    minX, minY, minZ = 10000000, 10000000, 10000000
    maxX, maxY, maxY = -10000000, -10000000, -10000000
    count = 0
    for line in open(filename):
        [x, y, z, vx, vy, vz] = [float(val) for val in line.split()]

        if (x < minX): minX = x
        if (y < minX): minX = y
        if (z < minX): minX = z

        if (x > maxX): maxX = x
        if (y > maxY): maxY = y
        if (z > maxZ): maxZ = z

        if (count % 1000000 == 0):
            print count/1000000, 'million particles scanned. current results:', minX, maxX, '\t', minY, maxY, '\t', minZ, maxZ
        count += 1


XX, YY, ZZ, VX, VY, VZ = range(6)
scan_min_max('sims/new_sim1_exact.txt')

"""
Creation of isolated cube data. 

18 divisions per axis -> 18**3 cubes,
17 divisions per axis -> 17**3 cubes; 
data from cube nearest the origin is retained.
"""
#isolate_particles(div_per_axis = 18, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_18.txt')
#isolate_particles(div_per_axis = 18, bindex = [0, 0, 0], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_18.txt')

#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_17.txt')
#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_17.txt')

exact_cube = np.loadtxt('sim1_partial_exact_18.txt')
approx_cube = np.loadtxt('sim1_partial_approx_18.txt')

print
print 'number of particles in exact cube:',len(exact_cube)
print 'number of particles in approx cube:',len(approx_cube)

print
print 'exact X range:',min(exact_cube[:,XX]),max(exact_cube[:,XX])
print 'approx X range:',min(approx_cube[:,XX]),max(approx_cube[:,XX])
print 'exact Y range:',min(exact_cube[:,YY]),max(exact_cube[:,YY])
print 'approx Y range:',min(approx_cube[:,YY]),max(approx_cube[:,YY])
print 'exact Z range:',min(exact_cube[:,ZZ]),max(exact_cube[:,ZZ])
print 'approx Z range:',min(approx_cube[:,ZZ]),max(approx_cube[:,ZZ])

data = [[exact_cube[:,XX]*10, approx_cube[:,XX]*10], [exact_cube[:,YY], approx_cube[:,YY]]]
labels = ['X Pos (kpc/10)', 'Y Pos (kpc)']
titles = ['Exact', 'Approx']
vis(data, labels, titles)

figure(0)
plot(exact_cube[:,XX], exact_cube[:,YY], '.')
axes = gca()
axes.set_ylim(-35, 0)

figure(1)
plot(approx_cube[:,XX], approx_cube[:,YY], '.')
axes = gca()
axes.set_ylim(-35, 0)



