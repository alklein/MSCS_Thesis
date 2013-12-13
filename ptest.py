#!/usr/bin/python2.7

from toy import *

def custom_pll_test():
    
    M = 1500
    eta = M

    print
    print ' > [debug] Making new toyData object with M, eta =',M,'...'
    tD = toyData(M = M, eta = eta)
    tD.make_samples()
    #tD.save_samples('data.txt', append=True)

    all_data = tD.all_samples
    train_data = tD.train_samples
    cv_data = tD.cv_samples
    test_data = tD.test_samples

    print
    print ' > [debug] Training estimator... '
    E = Estimator(train_data, cv_data, dist_fn = L2_distance, kernel = RBF_kernel)

    #E.train(parallel=False, verbose=True)    
    E.train(parallel=True, num_processes=20, verbose=True)


custom_pll_test()
