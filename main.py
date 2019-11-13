import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import itertools
import pickle

import pic.dirichlet_neumann

def main():
    run_tests()
    pic.dirichlet_neumann.dirichlet_neumann_test()
#end def main

def run_tests():
    import doctest
    verbose = False
    doctest.testmod(pic.grid, verbose=verbose)
    doctest.testmod(pic.particle, verbose=verbose)
    doctest.testmod(pic.distributions, verbose=verbose)
#end def run_tests

if __name__ == '__main__':
    main()
