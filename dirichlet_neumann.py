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

def dirichlet_neumann_test():
    #Imports and constants
    density = 1e18
    N = 20000
    timesteps = 1000
    ng = 400
    dt = 4e-10
    Ti = 10.*11600
    Te = 20.*11600
    floating_potential = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 300.*LD
    print(f'L: {L}')
    print(f'Phi: {floating_potential}')
    p2c = density*L/N
    print(f'ppc: {p2c}')
    alpha = 0.0*np.pi/180.0
    B0 = 0.
    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])

    E0 = 0.
    E = np.array([E0, E0, E0])

    #vx = -np.sqrt(kb*Te/mp)
    vx = 0.
    print(f'u0: {vx}')

    #Skip every nth plot
    skip = 1

    #Initialize objects, generators, and counters
    grid = Grid(ng//2, L/2, Te, bc='dirichlet-neumann')
    grid2 = Grid(ng, L, Te, bc = 'dirichlet-dirichlet')

    particles = [Particle(1.*mp, 1, p2c, Ti, Z=1, B0=B, E0=E, grid=grid, vx=vx) \
        for _ in range(N//2)]

    particles2 = [Particle(1.*mp, 1, p2c, Ti, Z=1, B0=B, E0=E, grid=grid2, vx=vx) \
        for _ in range(N)]

    vth = particles[0].vth
    print(f'vth: {vth}')

    source_distribution = source_distribution_6D(grid2, Ti, mp, vx=vx)
    #flux_distribution = flux_distribution_6D(grid, Ti, mp, vx=vx)

    plt.ion()
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)

    positions = np.zeros(N)
    velocities = np.zeros(N)

    positions2 = np.zeros(N)
    velocities2 = np.zeros(N)

    flux = 2e22
    num_particles_per_timestep = int(flux / p2c * dt)
    num_particles_per_timestep = N

    n_energy = 200
    n_angle = 90
    energy_bins = np.linspace(0.0, 100.0*e, n_energy + 1)
    angle_bins = np.linspace(0.0, 90.0, n_angle + 1)

    iead = np.zeros((n_energy, n_angle))
    iead2 = np.zeros((n_energy, n_angle))

    time = 0.
    for time_index in range(timesteps+1):
        energies = []
        angles = []
        energies2 = []
        angles2 = []

        time += dt
        print(f'timestep: {time_index}')
        print(f'n0: {grid.n0}\nadded_particles: {grid.added_particles}')

        grid.weight_particles_to_grid_boltzmann_neumann(particles, dt)
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_neumann_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()
        print(f'max phi: {np.max(grid.phi)}')

        grid2.weight_particles_to_grid_boltzmann(particles2, dt)
        grid2.n0 = grid.n0
        grid2.reset_added_particles()
        grid2.solve_for_phi_dirichlet_boltzmann()
        grid2.differentiate_phi_to_E_dirichlet()
        print(f'max phi 2: {np.max(grid2.phi)}')

        #Particle loop
        source_counter = 0
        for particle_index, particle in enumerate(particles):
            if particle.is_active():

                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt)

                particle.apply_BCs_dirichlet_neumann(grid)

                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x/particle.vth

                if not particle.is_active():
                    energies.append(particle.kinetic_energy)
                    angles.append(particle.get_angle_wrt_wall())

            elif source_counter < num_particles_per_timestep:
                particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1, 1)
                source_counter += 1
                if particle.r[0] > L/2:
                    particle.r[3] = -particle.r[3]
                    particle.r[0] = L/2 - (particle.r[0] - L/2)
                source_counter += 1
        #end for particle_index, particle

        #Particle loop 2
        source_counter = 0
        for particle_index, particle in enumerate(particles2):
            if particle.is_active():

                particle.interpolate_electric_field_dirichlet(grid2)
                particle.push_6D(dt)

                particle.apply_BCs_dirichlet(grid2)

                positions2[particle_index] = particle.x
                velocities2[particle_index] = particle.v_x/particle.vth

                if not particle.is_active():
                    energies2.append(particle.kinetic_energy)
                    angles2.append(particle.get_angle_wrt_wall())

            elif source_counter < num_particles_per_timestep:
                particle.reactivate(source_distribution, grid2, time, p2c, 1.*mp, 1, 1)
        #end for particle_index, particle

        plt.figure(1)
        plt.clf()
        plt.scatter(positions, velocities, s=0.1, c='black')
        plt.scatter(positions2, velocities2, s=0.1, c='red')
        plt.axis([0.0, grid2.length, -6. + vx/vth, 6. + vx/vth])
        plt.pause(0.01)

        plt.figure(2)
        plt.clf()
        iead_temp, _, _ = np.histogram2d(energies, angles, (energy_bins, angle_bins))
        print(np.shape(iead_temp))
        iead += iead_temp
        plt.pcolormesh(angle_bins, energy_bins, iead)
        plt.pause(0.01)

        plt.figure(3)
        plt.clf()
        iead_temp, _, _ = np.histogram2d(energies2, angles2, (energy_bins, angle_bins))
        iead2 += iead_temp
        plt.pcolormesh(angle_bins, energy_bins, iead2)
        plt.pause(0.01)

        plt.figure(4)
        plt.clf()
        plt.plot(np.linspace(0.0, L/2, ng//2), grid.phi, c='black')
        plt.plot(np.linspace(0.0, L/2, ng//2), grid2.phi[:ng//2], c='red')
        plt.pause(0.01)

    breakpoint()
#end def pic_bca
