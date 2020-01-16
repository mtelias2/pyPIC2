from pic.particle import *
from pic.grid import *
from pic.distributions import *
from io import *
from constants.constants import *
from fractal_tridyn.utils.generate_ftridyn_input import *

def pic_iead():
    import generate_ftridyn_input as gen
    density = 1e20
    densities_boron = [1e11, 1e12, 1e12, 1e11, 1e13]
    N = 1000
    timesteps = 10
    ng = 600
    dt = 1e-10
    Ti = 10.*11600.
    Te = 10.*11600.
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 300.*LD
    p2c = density*L/N
    p2cs_boron = [density_boron*L/N for density_boron in densities_boron]
    alpha = 86.*np.pi/180.
    B0 = np.array([2.*np.cos(alpha), 2.*np.sin(alpha), 0.])
    E0 = np.zeros(3)
    number_histories = 100
    num_energies = 40
    num_angles = 40

    phi_floating = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    print(f'Floating potential: {phi_floating} V')

    #Initialize objects, generators, and counters
    grid = Grid(ng, L, Te, bc='dirichlet-dirichlet')

    deuterium = [
        Particle(2.*mp, e, p2c, Ti, Z=1, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    boron_1 = [
        Particle(10.81*mp, e, p2cs_boron[0], Ti, Z=5, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    boron_2 = [
        Particle(10.81*mp, 2.*e, p2cs_boron[1], Ti, Z=5, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    boron_3 = [
        Particle(10.81*mp, 3.*e, p2cs_boron[2], Ti, Z=5, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    boron_4 = [
        Particle(10.81*mp, 4.*e, p2cs_boron[3], Ti, Z=5, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    boron_5 = [
        Particle(10.81*mp, 5.*e, p2cs_boron[4], Ti, Z=5, B0=B0, E0=E0, grid=grid)
        for _ in range(N)
    ]

    source_distribution = source_distribution_6D(grid, Ti, mp)
    impurity_distribution = source_distribution_6D(grid, Ti, 10.81*mp)

    particles = deuterium + boron_1 + boron_2 + boron_3 + boron_4 + boron_5

    N = len(particles)

    #tridyn_interface_D_B = gen.tridyn_interface('D', 'B')
    #tridyn_interface_wall_wall = gen.tridyn_interface('B', 'B')

    tridyn_interface_D_B = tridyn_interface('D', 'B')
    tridyn_interface_wall_wall = tridyn_interface('B', 'B')

    iead_D = np.zeros((num_energies, num_angles))
    iead_B = np.zeros((num_energies, num_angles))

    skip = 1

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)
    #plt.ion()

    #Start of time loop
    time = 0.
    for time_index in range(timesteps+1):
        #Clear iead collection arrays
        energies_D = []
        angles_D = []
        energies_B = []
        angles_B = []
        #TODO ADD MORE BORON CHARGE STATES?

        #Clear plotting arrays
        positions = np.zeros(N)
        velocities = np.zeros(N)
        colors = np.zeros(N)

        time += dt
        grid.weight_particles_to_grid_boltzmann(particles, dt)
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        #Print parameters to command line
        print(f'timestep: {time_index}')
        print(f'n0: {grid.n0}\nadded_particles: {grid.added_particles}')
        print(f'phi_max: {np.max(grid.phi)}')

        #Begin particle loop
        for particle_index, particle in enumerate(particles):
            #If particle is active, push particles and store positions, velocities
            if particle.is_active():
                #Store particle coordinates for plotting
                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x
                colors[particle_index] = particle.charge_state

                #Interpolate E, push in time, and apply BCs
                particle.interpolate_electric_field(grid)
                particle.push_6D(dt)
                particle.apply_BCs_dirichlet(grid)

                #If particle is deactivated at wall, store in iead colleciton arrays
                if not particle.is_active():
                    if particle.Z == 1:
                        energies_D.append(particle.kinetic_energy/e)
                        angles_D.append(particle.get_angle_wrt_wall())
                    #end if
                    if particle.Z == 5:
                        energies_B.append(particle.kinetic_energy/e)
                        angles_B.append(particle.get_angle_wrt_wall())
                    #end if
            #If particle is not active, reinitialize as either source H or impurity B
            else:
                if np.random.choice((True, True), p=(1./6., 5./6.)):
                    particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1.*e, 1)
                else:
                    charge_state = np.random.choice((1,2,3,4,5))
                    particle.reactivate(impurity_distribution, grid, time, p2cs_boron[charge_state-1], 10.81*mp, charge_state*e, 5)
            #end if
        #end for particle_index, particle

        #Collect iead arrays into 2D IEAD histogram
        iead_D_temp, energy_range, angle_range = np.histogram2d(energies_D, angles_D, bins=(num_energies, num_angles), range=[[0., 4.*phi_floating], [0., 90.]])
        iead_B_temp, energy_range, angle_range = np.histogram2d(energies_B, angles_B, bins=(num_energies, num_angles), range=[[0., 4.*phi_floating], [0., 90.]])
        iead_D += iead_D_temp
        iead_B += iead_B_temp


        #Plotting routine
        if time_index%skip == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(grid.domain, grid.phi)
            plt.draw()
            plt.savefig('pic_bca_phi'+str(time_index))
            plt.pause(0.001)

            plt.figure(2)
            plt.clf()
            plt.scatter(positions, velocities, s=0.5, c=colors-1., cmap='viridis')
            plt.axis((0., L, -6.0*particles[0].vth, 6.0*particles[0].vth))
            plt.draw()
            plt.savefig('pic_bca_ps'+str(time_index))
            plt.pause(0.001)

            plt.figure(3)
            plt.clf()
            plt.pcolormesh(angle_range, energy_range, iead_D.transpose())
            plt.draw()
            plt.pause(0.001)

            plt.figure(4)
            plt.clf()
            plt.pcolormesh(angle_range, energy_range, iead_B.transpose())
            plt.draw()
            plt.pause(0.001)
        #end if
    #end for time_index
    plt.figure(3)
    plt.savefig('iead_D')
    plt.figure(4)
    plt.savefig('iead_B')
    new_particle_list_D_s, new_particle_list_D_r = tridyn_interface_D_B.run_tridyn_simulations_from_iead(energy_range, angle_range, iead_D, number_histories=number_histories)
    new_particle_list_B_s, new_particle_list_B_r = tridyn_interface_wall_wall.run_tridyn_simulations_from_iead(energy_range, angle_range, iead_B, number_histories=number_histories)
    num_incident_B = np.sum(iead_B)
    num_deposited_B = np.sum(iead_B) - len(new_particle_list_B_r)//number_histories
    num_reflected_B = len(new_particle_list_B_r)//number_histories
    num_sputtered_B = len(new_particle_list_B_s)//number_histories + len(new_particle_list_D_s)//number_histories
    print(f'num_deposited: {num_deposited_B}, num_sputtered: {num_sputtered_B}, {num_reflected_B}, {num_incident_B}')
