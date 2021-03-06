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

from pylab import *
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
            print 'particles found so far:', format(len(ps), ',d')
        count += 1

        cur_p = [float(val) for val in line.split()]
        x, y, z = cur_p[0], cur_p[1], cur_p[2]
        if ((inner_x < x) and (x <= outer_x) and (inner_y < y) and (y <= outer_y) and (inner_z < z) and (z <= outer_z)): 
            ps.append(cur_p)
    
    return np.array(ps)

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

def bns_ext(data, res=400, xmin=None, xmax=None, ymin=None, ymax=None):
    if (not xmin or not xmax or not ymin or not ymax):
        xmins, xmaxs = map(min, data[0]), map(max, data[0])
        ymins, ymaxs = map(min, data[1]), map(max, data[1])
        xmin, xmax = min(xmins), max(xmaxs)
        ymin, ymax = min(ymins), max(ymaxs)
    xrnge, yrnge = np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res)
    return [yrnge, xrnge], [xmin, xmax, ymin, ymax]

# description: makes colored 'scatter' plots for side-by-side comparison of data
# input:
     # data = [[ x1, x2, ... ],[ y1, y2, ... ]]
     # labels = [ xlabel, ylabel ]
     # titles = [ title1, title2, ... ]
# output: n/a (makes graph)
# effects: n/a
# notes: does not take log of data (must use np.log beforehand for semilog or log-log vis)

def vis(data, labels, titles, res=400, vmax=10.0, xmin=None, xmax=None, ymin=None, ymax=None):
    i = 0
    while i < len(data[0]):
        figure(i)
        xs, ys = data[0][i], data[1][i]
        bns, ext = bns_ext(data, res, xmin, xmax, ymin, ymax)
        H, yedges, xedges = np.histogram2d(ys, xs, bins=bns)
        imshow(H, extent=ext, aspect='equal', cmap = cm.jet, interpolation='nearest', origin='lower', vmin=0.0, vmax=vmax)
        colorbar()
        rc('text', usetex=True)
        xlabel(labels[0], fontsize=20)
        ylabel(labels[1], fontsize=20)
        title(titles[i], fontsize=24)
        i += 1
    show()

def scan_min_max(filename):
    minX, minY, minZ = 10000000, 10000000, 10000000
    maxX, maxY, maxZ = -10000000, -10000000, -10000000
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

"""
Returns empirical min and max values of an entire dataset
along some axis (the specified column). Uses constant memory.
"""
def lowmem_global_min_max(filename, col, verbose=False, start_min = 1000000, start_max = -1000000):
    if (verbose): print '\n >>> Finding global minmax of',filename,'column',col,'\n'
    cur_min, cur_max = start_min, start_max
    count = 0
    for line in open(filename):
        cur_p = [float(val) for val in line.split()]
        cur_val = cur_p[col]
        if (cur_val < cur_min): cur_min = cur_val
        if (cur_val > cur_max): cur_max = cur_val
        if (count % 10000000 == 0): 
            print count/1000000,'- cur min:',cur_min,'- cur max:',cur_max
        count += 1
    return (cur_min, cur_max)

"""
Returns values in file called filename as array of arrays of floats
"""
def load_floats(filename):
    result = []
    for line in open(filename):
        result.append([float(val) for val in line.split()])
    return np.array(result)

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
Maps bindices to their particles in 3D.
"""
def assign_particles_3D_fast(filename, bindices, xmin, ymin, zmin, binsz_x, binsz_y, binsz_z, num_bins, verbose=False, chunk=1000000):

    assignments = {str(bindex) : [] for bindex in bindices}
    count = 0

    for line in open(filename):

        if ((count % chunk == 0) and (verbose)):
            print count/chunk,'million particles assigned...'
        count += 1

        cur_p = [float(val) for val in line.split()]
        sc_x = (cur_p[0] - xmin)/(binsz_x)
        sc_y = (cur_p[1] - ymin)/(binsz_y)
        sc_z = (cur_p[2] - zmin)/(binsz_z)
        cur_bindex = [int(sc_x), int(sc_y), int(sc_z)]
        if (str(cur_bindex) in assignments):
            assignments[str(cur_bindex)].append(cur_p)

    return assignments                


def examine_binned_sample():
    
    infile = 'sim1_partial_approx_18_111.txt'
    print ' >>> ASSIGNING PARTICLES FROM',infile,'TO BINS <<<'

    (xmin, xmax) = constants.exact_col_0_min_max
    (ymin, ymax) = constants.exact_col_1_min_max
    (zmin, zmax) = constants.exact_col_2_min_max

    div_per_axis = 18
    
    binsz_x = (xmax - xmin)/div_per_axis
    binsz_y = (ymax - ymin)/div_per_axis
    binsz_z = (zmax - zmin)/div_per_axis

    new_xmin = xmin + binsz_x
    new_ymin = ymin + binsz_y
    new_zmin = zmin + binsz_z

    divs = [14, 15] #, 30, 40, 50, 60, 70, 80, 90, 100]
    for new_div_per_axis in divs:

        new_binsz_x = binsz_x/new_div_per_axis
        new_binsz_y = binsz_y/new_div_per_axis
        new_binsz_z = binsz_z/new_div_per_axis

        print '\n >>> NEW EXPERIMENT'
        print ' >>> >>> divisions per dimension:', new_div_per_axis
        print ' >>> >>> xmin:', new_xmin
        print ' >>> >>> binsz_x:', new_binsz_x,'\n'

        new_bindices = bindices_3D(new_div_per_axis)    
        assignments = assign_particles_3D_fast(infile, new_bindices, new_xmin, new_ymin, new_zmin, new_binsz_x, new_binsz_y, new_binsz_z, new_div_per_axis, verbose=True)
        counts = [len(assignments[key]) for key in assignments]
        print '... total num bins:', new_div_per_axis**3
        print '... avg. num particles per bin:',np.average(counts)
        print '... num empty bins:',len([c for c in counts if c < 1])

        figure(new_div_per_axis)
        hist(counts, bins=50, log=True)
        xlabel('Count', fontsize=24)
        ylabel('Number', fontsize=24)

    show()


XX, YY, ZZ, VX, VY, VZ = range(6)
examine_binned_sample()

#lowmem_global_min_max('sims/new_sim1_exact.txt', col=0, verbose=True)

#lowmem_global_min_max('sims/new_sim1_approx.txt', col=0, verbose=True)
#lowmem_global_min_max('sims/new_sim1_approx.txt', col=1, verbose=True)
#lowmem_global_min_max('sims/new_sim1_approx.txt', col=2, verbose=True)

"""
Creation of isolated cube data. 

18 divisions per axis -> 18**3 cubes,
17 divisions per axis -> 17**3 cubes; 
data from cube nearest the origin is retained.
"""
#isolate_particles(div_per_axis = 18, bindex = [1, 1, 1], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_18_111.txt')
#isolate_particles(div_per_axis = 18, bindex = [1, 1, 1], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_18_111.txt')

#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_exact.txt', outfile = 'sim1_partial_exact_17.txt')
#isolate_particles(div_per_axis = 17, bindex = [0, 0, 0], infile = 'sims/new_sim1_approx.txt', outfile = 'sim1_partial_approx_17.txt')

print '\nloading exact cube...'
#exact_cube = np.loadtxt('sim1_partial_exact_18_111.txt')
exact_cube = load_floats('sim1_partial_exact_18_111.txt')
print ' > length:',len(exact_cube)
print '\nloading approx cube...'
#approx_cube = np.loadtxt('sim1_partial_approx_18_111.txt')
approx_cube = load_floats('sim1_partial_approx_18_111.txt')
print ' > length:',len(approx_cube)

print
print 'number of particles in exact cube:',len(exact_cube)
print 'number of particles in approx cube:',len(approx_cube)

print
print '>>> empirical ranges:'
print 'exact X range:',min(exact_cube[:,XX]),max(exact_cube[:,XX])
print 'approx X range:',min(approx_cube[:,XX]),max(approx_cube[:,XX])
print 'exact Y range:',min(exact_cube[:,YY]),max(exact_cube[:,YY])
print 'approx Y range:',min(approx_cube[:,YY]),max(approx_cube[:,YY])
print 'exact Z range:',min(exact_cube[:,ZZ]),max(exact_cube[:,ZZ])
print 'approx Z range:',min(approx_cube[:,ZZ]),max(approx_cube[:,ZZ])

(xmin, xmax) = constants.exact_col_0_min_max
(ymin, ymax) = constants.exact_col_1_min_max
(zmin, zmax) = constants.exact_col_2_min_max

div_per_axis = 18

binsz_x = (xmax - xmin)/div_per_axis
binsz_y = (ymax - ymin)/div_per_axis
binsz_z = (zmax - zmin)/div_per_axis

print
print '>>> boundaries of cube:'
print 'xmin, xmax:',xmin + binsz_x,xmin + 2*binsz_x
print 'ymin, ymax:',ymin + binsz_y,ymin + 2*binsz_y
print 'zmin, zmax:',zmin + binsz_z,zmin + 2*binsz_z

zcut_in, zcut_out = zmin + binsz_z, zmin + binsz_z + binsz_z/100.
print
print '>>> cutting data along z axis to lie between',zcut_in,'and',zcut_out
print 'total length of exact cube:',len(exact_cube)
exact_cube_bottom = np.array([p for p in exact_cube if p[2] <= zcut_out])
print 'length of exact cube bottom:',len(exact_cube_bottom)
print 'total length of approx cube:',len(approx_cube)
approx_cube_bottom = np.array([p for p in approx_cube if p[2] <= zcut_out])
print 'length of approx cube bottom:',len(approx_cube)

zcut_in, zcut_out = zmin + 2*binsz_z - binsz_z/100., zmin + 2*binsz_z 
print
print '>>> cutting data along z axis to lie between',zcut_in,'and',zcut_out
print 'total length of exact cube:',len(exact_cube)
exact_cube_top = np.array([p for p in exact_cube if p[2] >= zcut_in])
print 'length of exact cube top:',len(exact_cube_top)
print 'total length of approx cube:',len(approx_cube)
approx_cube_top = np.array([p for p in approx_cube if p[2] >= zcut_in])
print 'length of approx cube top:',len(approx_cube_top)

zcut_in, zcut_out = zmin + binsz_z + binsz_z/2., zmin + binsz_z + binsz_z/2. + binsz_z/100.
print
print '>>> cutting data along z axis to lie between',zcut_in,'and',zcut_out
print 'total length of exact cube:',len(exact_cube)
exact_cube_middle = np.array([p for p in exact_cube if p[2] >= zcut_in and p[2] <= zcut_out])
print 'length of exact cube middle:',len(exact_cube_middle)
print 'total length of approx cube:',len(approx_cube)
approx_cube_middle = np.array([p for p in approx_cube if p[2] >= zcut_in and p[2] <= zcut_out])
print 'length of approx cube middle:',len(approx_cube_middle)

data = [[exact_cube[:,XX], approx_cube[:,XX]], [exact_cube[:,YY], approx_cube[:,YY]]]
labels = ['X Coordinate', 'Y Coordinate']
titles = ['', '']
vis(data, labels, titles, res=500, vmax=10.0, xmin = xmin + binsz_x, xmax = xmin + 2*binsz_x, ymin = ymin + binsz_y, ymax = ymin + 2*binsz_y)

data = [[exact_cube_bottom[:,XX], approx_cube_bottom[:,XX], exact_cube_top[:,XX], approx_cube_top[:,XX], exact_cube_middle[:,XX], approx_cube_middle[:,XX]], \
            [exact_cube_bottom[:,YY], approx_cube_bottom[:,YY], exact_cube_top[:,YY], approx_cube_top[:,YY], exact_cube_middle[:,YY], approx_cube_middle[:,YY]]]
labels = ['X Coordinate', 'Y Coordinate']
#titles = ['Exact (bottom)', 'Approx (bottom)', 'Exact (top)', 'Approx (top)', 'Exact (middle)', 'Approx (middle)']
titles = ['', '', '', '', '', '']
vis(data, labels, titles, res=500, vmax=10.0, xmin = xmin + binsz_x, xmax = xmin + 2*binsz_x, ymin = ymin + binsz_y, ymax = ymin + 2*binsz_y)

figure(0)
plot(exact_cube[:,XX], exact_cube[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

figure(1)
plot(approx_cube[:,XX], approx_cube[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

figure(2)
plot(exact_cube_bottom[:,XX], exact_cube_bottom[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

figure(3)
plot(approx_cube_bottom[:,XX], approx_cube_bottom[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

figure(4)
plot(exact_cube_top[:,XX], exact_cube_top[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

figure(5)
plot(approx_cube_top[:,XX], approx_cube_top[:,YY], '.')
axes = gca()
axes.set_xlim(xmin + binsz_x, xmin + 2*binsz_x)
axes.set_ylim(ymin + binsz_y, ymin + 2*binsz_y)
xlabel('X Coordinate', fontsize=20)
ylabel('Y Coordinate', fontsize=20)

show()
