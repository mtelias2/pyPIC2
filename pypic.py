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

from particle import *
from grid import *
from distributions import *
from convert import *
from constants import *
from pic_bca import *
from dirichlet_neumann import *

def main():
    run_tests()
    dirichlet_neumann_test()
#end def main

def run_tests():
    import doctest
    doctest.testmod()
#end def run_tests

if __name__ == '__main__':
    main()
