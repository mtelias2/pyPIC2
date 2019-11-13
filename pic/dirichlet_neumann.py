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

from pic.particle import *
from pic.grid import *
from pic.distributions import *
#from io.convert import *
from constants.constants import *

def dirichlet_neumann_test():
    #User inputs
    density = 1e18
    Te = 10.*11600.
    Ti = 10.*11600.
    ppc = 100
    ng_per_debye = 3
    nt_per_oscillation = 5
    n_ion_transit_times = 5.0
    num_debye = 1000
    alpha = 0.0*np.pi/180.0
    B0 = 0.
    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])
    E0 = 0.
    E = np.array([E0, E0, E0])
    vx = 0.
    print(f'u0: {vx}')

    #Skip every nth plot
    skip = 1

    source = 1000

    #Physical parameters
    floating_potential = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    ion_plasma_frequency = np.sqrt(density*e**2/mp/epsilon0)/2./np.pi

    #Numerical parameters
    N = ppc*ng_per_debye*num_debye
    L = num_debye*LD
    dt = 1./ion_plasma_frequency/nt_per_oscillation
    ng = num_debye*ng_per_debye
    p2c = density*L/N
    ion_thermal_velocity = np.sqrt(kb*Ti/mp)
    ion_transit_time = L/ion_thermal_velocity
    timesteps = int(n_ion_transit_times * ion_transit_time / dt)

    print(f'timesteps: {timesteps}')
    print(f'Source: {source} particles per timestep')
    print(f'dt: {dt}')
    print(f'N: {N}')
    print(f'ng: {ng}')
    print(f'L: {L}')
    print(f'floating potential: {floating_potential}')

    #Initialize objects, generators, and counters
    grid = Grid(ng, L, Te, bc='dirichlet-neumann')

    particles = [Particle(1.*mp, 1, p2c, Ti, Z=1, B0=B, E0=E, grid=grid, vx=vx) \
        for _ in range(N)]

    vth = particles[0].vth
    print(f'vth: {vth}')

    source_distribution = source_distribution_6D(grid, Ti, mp, vx=vx)
    flux_distribution = flux_distribution_6D(grid, Ti, mp, vx=vx)

    plt.ion()
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)

    positions = np.zeros(N)
    velocities = np.zeros(N)

    time = 0.
    for time_index in range(timesteps+1):
        velocities_boundary = []

        time += dt
        print(f'timestep: {time_index}')
        print(f'n0: {grid.n0}\nadded_particles: {grid.added_particles}')
        grid.weight_particles_to_grid_boltzmann(particles, dt)
        #grid.smooth_rho()
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_neumann_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        #Begin particle loop

        source_counter = 0

        velocities_at_boundary = [particle.v_x for particle in particles if (grid.length - 3*grid.dx <  particle.x < grid.length)]

        for particle_index, particle in enumerate(particles):
            if particle.is_active():

                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt)

                #particle.apply_BCs_dirichlet_neumann(grid)
                particle.apply_BCs_dirichlet_reflection(grid)

                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x/particle.vth

                if grid.length - 3.*grid.dx < particle.x < grid.length:
                    velocities_boundary.append(particle.v_x)

            elif source_counter < source:
                #particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1.*e, 1)
                particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1, 1)
                #new_velocity = sample_to_fill_distribution(gaussian_distribution, (vx, vth), -6*vth, 6*vth, 200, velocities_at_boundary, 1)
                #particle.r[3] = -np.abs(particle.r[3])
                source_counter += 1

        #end for particle_index, particle
        print(f'Particles from right wall: {source_counter}')

        plt.figure(1)
        plt.clf()
        plt.scatter(positions, velocities, s=0.1, c='black')
        plt.axis([0.0, grid.length, -6. + vx/vth, 6. + vx/vth])
        plt.pause(0.01)

        plt.figure(2)
        plt.clf()
        plt.plot(np.linspace(0.0, grid.length, grid.ng), grid.phi)
        plt.pause(0.01)

        plt.figure(3)
        plt.clf()
        plt.plot(np.linspace(0.0, grid.length, grid.ng), grid.rho/e)
        plt.pause(0.01)

        plt.figure(4)
        plt.clf()
        heights, bins, _ = plt.hist(velocities_boundary, bins = np.linspace(-9*vth, 9*vth, 100), density=True)
        dist = [2*gaussian_distribution(bin, vx, vth) for bin in bins]
        #dist /= np.max(dist)
        plt.plot(bins, dist)
        plt.pause(0.01)

    breakpoint()
#end def pic_bca
