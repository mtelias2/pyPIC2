import numpy as np

from constants import *
from pic.grid import *
from pic.particle import *

def source_distribution_6D(grid, Ti, mass, vx=0.):
    '''
    This generator produces an iterable that samples from a Maxwell-
    Boltzmann distribution for a given temperature and mass for velocities, and
    uniformly samples the grid for positions. y and z positions are started as
    0.0.

    Args:
        grid (Grid): grid object where the particles will be (re)initialized
        Ti (float): temperature of species being sampled
        mass (float): mass of species being sampled

    Yields:
        r (ndarray): 7-element particle coordinate array in 6D coordinates

    Tests:
        >>> grid = Grid(100, 1.0, 1.0)
        >>> distribution = source_distribution_6D(grid, 1.0, 1.0)
        >>> r = next(distribution)
        >>> 0.0 < r[0] < 1.0
        True
    '''
    while True:
        vth = np.sqrt(kb*Ti/mass)
        r = np.empty(7)
        #r[0] = np.random.uniform(0.0, grid.length)
        #r[0] = np.random.normal(grid.length/2, grid.length/12.0)
        #r[0] %= grid.length
        r[0] = np.random.uniform(0., grid.length)
        r[1:3] = 0.
        r[3:6] = np.random.normal(0.0, vth, 3) + vx
        yield r
    #end while
#end def source_distribution_6D

def weighted_gaussian(x, mu, sigma):
    '''
    This function is a Gaussian weighted by |x|. This is used to maintain a
    maxwellian distribution at an open boundary of a particle-in-cell
    simulation, since particles with less x-velocity will spend more time closer
    to the wall, proportional to their inverse speed.

    Args:
        x (float): x-value
        mu (float): mean value
        sigma (float) standard deviation

    Returns:
        y (float): height of distribution
    '''
    return gaussian_distribution(x, mu, sigma)*np.abs(x)

def flux_distribution_6D(grid, Ti, mass, vx=0., gamma=0., vx_pert=0.):
    while True:
        vth = np.sqrt(kb*Ti/mass)
        r = np.empty(7)
        #r[0] = np.random.uniform(0.0, grid.length)
        r[0] = grid.length - grid.dx * np.random.uniform(0., 1.)
        r[1:3] = 0.
        r[3:6] = np.random.normal(0.0, vth, 3)
        num_vels = 100
        vels = np.linspace(-6*vth, 6*vth, num_vels)
        dist = np.array([weighted_gaussian(vel, vx, vth) for vel in vels])
        dist[vels>=0] = 0.
        dist /= np.sum(dist)
        r[3] = -np.abs(np.random.choice(vels, p=dist)) + np.random.uniform(-1, 1)*(vels[1] - vels[0])/2.
        #r[3] += vx
        if np.random.uniform(0, 1) < gamma:
            r[3] = vx_pert*vth
        yield r
    #end while
#end def flux_distribution_6D

def sample_to_fill_distribution(ideal_distribution, dist_args, min, max, num_bins, population, sample_size=1):
    '''
    Given an ideal distribution, this function samples from the clipped and
    normalized difference of the two distributions. This should allow one to
    maintain a specific distribution shape if components of it are being lost
    with some bias.

    Args:
        ideal_distribution (function): distribution to attempt to match
        dist_args (list): additional arguments to ideal_distribution
        min (float): minimum x-value of distribution.
        max (float): maximum x-value of distribution.
        num_bins (int): number of bins
        population (list): population to add to via sampling
        sample_size (int): number of samples to return.

    Returns:
        new_population (float(s)): new sample(s) from clipped difference
    '''
    heights, bins = np.histogram(population, bins=np.linspace(min, max, num_bins), density=True)
    centers = bins[:-1] + (bins[1] - bins[0])/2.
    ideal_distribution_heights = ideal_distribution(centers, *dist_args)
    difference = np.clip(ideal_distribution_heights - heights, 0, None)
    difference /= np.sum(difference)

    new_population = np.random.choice(centers, size=sample_size, p=difference)
    new_population += np.random.uniform(-1, 1, sample_size)*(centers[1] - centers[0])

    return new_population

def gaussian_distribution(x, mu, sigma):
    '''
    Gaussian distribution.

    Args:
        x (float): x-value of distribution
        mu (float): mean value of distribution
        sigma (float): standard deviation of distribution

    Returns:
        y (float): distribution height at x
    '''
    return 1./np.sqrt(2.*np.pi*sigma**2)*np.exp(-(x - mu)**2/(2.*sigma**2))
