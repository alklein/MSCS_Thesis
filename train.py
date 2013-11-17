import sys
import time
import itertools
import numpy as np

from pylab import *
from Queue import Queue
from multiprocessing import Pool
from multiprocessing.managers import BaseManager
from toy import fourier_coeffs, cosine_basis, coeffs_to_approx_density

def load_samples(filename, max_count=5000):
  data = []
  count = 0
  for instance in open(filename):
    raw_inp, raw_outp = instance.split(';')
    inp  = [float(val) for val in raw_inp.split(',')]
    outp = [float(val) for val in raw_inp.split(',')]
    data.append([inp, outp])
    count += 1
    if (count >= max_count): break
  return np.array(data)


def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i+n]

def Map(X):
  result = []
  for item in X:
    x, y = item[0], item[1]
    fx = fourier_coeffs(x, 20)
    fy = fourier_coeffs(y, 20)
    result.append([fx, fy])
  return result

def Reduce(XX):
  q = Queue()
  for item in XX:
    q.put(1)
  return q

data_file = 'data.txt'
print 'Loading Data'
data = load_samples(data_file) # TODO: load in parallel?
print ' > dimensions of data:',len(data),len(data[0]),len(data[0][0])

print ' > computing coeffs sequentially for comparison'
start_time = time.clock()
coeffs = Map(data)
diff = time.clock() - start_time
print ' > time to compute coeffs:',diff

ex_sample = data[0][0]
ex_coeffs = coeffs[0][0]
f_hat = coeffs_to_approx_density(ex_coeffs)
xs = np.array(range(100))/100.
figure(0)
hist(ex_sample, bins=100, normed=True, color='r')
plot(xs, map(f_hat, xs), linewidth=2, color='b')


for num_processes in [1, 2, 5, 10]:

  partitioned_data = list(chunks(data, len(data) / num_processes))
  P = Pool(processes=num_processes,)
  D = {}

  print
  print 'Dispatching Coefficient Computation'
  print ' > using',num_processes,'process(es)'
  start_time = time.clock()
  coeffs = P.map(Map, partitioned_data)
  P.close()
  P.join()
  diff = time.clock() - start_time
  print ' > time to compute coeffs:',diff

  coeffs = list(itertools.chain(*coeffs)) # flatten coeffs
  print ' > dimensions of flattened coeffs:',len(coeffs),len(coeffs[0]),len(coeffs[0][0])

print
print 'Coalescing Results'
# TODO: replace queue with ball tree
Q = Queue()
for [xcoeffs, ycoeffs] in coeffs:
  Q.put(str(xcoeffs))
  D[str(xcoeffs)] = str(ycoeffs)
print 'Final number of items in data structure:',Q.qsize()
