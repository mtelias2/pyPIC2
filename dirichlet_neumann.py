from particle import *
from grid import *
from distributions import *
from convert import *
from constants import *

def dirichlet_neumann_test():
    #Imports and constants
    density = 1e19
    N = 50000
    timesteps = 1000
    ng = 800
    dt = 2e-10
    Ti = 10.*11600
    Te = 10.*11600
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 400.*LD
    print(f'L: {L}')
    p2c = density*L/N
    alpha = 0.0*np.pi/180.0
    B0 = 0.
    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])

    E0 = 0.
    E = np.array([E0, E0, E0])

    vx = 0.0

    #Skip every nth plot
    skip = 1

    #Initialize objects, generators, and counters
    grid = Grid(ng, L, Te, bc='dirichlet-neumann')

    particles = [Particle(1.*mp, 1, p2c, Ti, Z=1, B0=B, E0=E, grid=grid, vx=vx) \
        for _ in range(N)]

    for particle in particles:
        particle.r[3]  = -np.abs(particle.r[3])

    #for particle in particles:
    #    particle.active = False

    vth = particles[0].vth

    source_distribution = source_distribution_6D(grid, Ti, mp, vx=vx)
    flux_distribution = flux_distribution_6D(grid, Ti, mp, vx=vx, gamma=0.5, vx_pert=-3.)

    plt.ion()
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)

    positions = np.zeros(N)
    velocities = np.zeros(N)

    flux = 100

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

        flux_counter = 0

        velocities_at_boundary = [particle.v_x for particle in particles if (grid.length - 3*grid.dx <  particle.x < grid.length)]

        for particle_index, particle in enumerate(particles):
            if particle.is_active():

                particle.interpolate_electric_field_dirichlet(grid)
                particle.push_6D(dt)

                #particle.apply_BCs_dirichlet_neumann(grid)
                particle.apply_BCs_dirichlet(grid)

                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x/particle.vth

                if grid.length - 3.*grid.dx < particle.x < grid.length:
                    velocities_boundary.append(particle.v_x)

            elif flux_counter < flux:
                #particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1.*e, 1)
                particle.reactivate(flux_distribution, grid, time, p2c, 1.*mp, 1, 1)
                #new_velocity = sample_to_fill_distribution(gaussian_distribution, (vx, vth), -6*vth, 6*vth, 200, velocities_at_boundary, 1)
                particle.r[3] = -np.abs(particle.r[3])
                flux_counter += 1

        #end for particle_index, particle

        plt.figure(1)
        plt.clf()
        plt.scatter(positions, velocities, s=0.1, c='black')
        plt.axis([0.0, grid.length, -6. + vx/vth, 6. + vx/vth])
        plt.draw()
        plt.pause(0.01)

        plt.figure(2)
        plt.clf()
        plt.plot(np.linspace(0.0, grid.length, grid.ng), grid.phi)
        plt.draw()
        plt.pause(0.01)

        plt.figure(3)
        plt.clf()
        plt.plot(np.linspace(0.0, grid.length, grid.ng), grid.rho/e)
        plt.draw()
        plt.pause(0.01)

        plt.figure(4)
        plt.clf()
        heights, bins, _ = plt.hist(velocities_boundary, bins = np.linspace(-9*vth, 9*vth, 100), density=True)
        dist = [2*gaussian_distribution(bin, vx, vth) for bin in bins]
        #dist /= np.max(dist)
        plt.plot(bins, dist)
        plt.draw()
        plt.pause(0.01)

    breakpoint()
#end def pic_bca
