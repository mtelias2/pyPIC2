from pic.particle import *
from pic.grid import *
from pic.distributions import *
from io import *
from constants.constants import *
from fractal_tridyn.utils.generate_ftridyn_input import *

def pic_bca():
    #Imports and constants
    import fractal_tridyn.utils.generate_ftridyn_input as gen
    density = 1e19
    N = 10000
    source_N = N
    timesteps = 1000
    ng = 150
    dt = 1e-10
    Ti = 50.*11600
    Te = 60.*11600
    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = 100.*LD
    print(f'L: {L}')
    p2c = density*L/N
    alpha = 86.0*np.pi/180.0
    B0 = 2.
    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])
    cross_section = 1e-15

    E0 = 0.
    E = np.array([E0, E0, E0])

    number_histories = 100
    num_energies = 25
    num_angles = 20
    iead = np.zeros((num_energies, num_angles))

    #Skip every nth plot
    skip = 1

    #Calculate floating potential
    phi_floating = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    print(f'Floating potential: {phi_floating} V')

    #Initialize objects, generators, and counters
    grid = Grid(ng, L, Te, bc='dirichlet-dirichlet')

    particles = [Particle(1.*mp, 1, p2c, Ti, Z=1, B0=B, E0=E, grid=grid) \
        for _ in range(N)]

    #particles += impurities
    #tridyn_interface = gen.tridyn_interface('H', 'B')
    #tridyn_interface_B = gen.tridyn_interface('B', 'B')

    tridyn_interface = tridyn_interface('H', 'B')
    tridyn_interface_B = tridyn_interface('B', 'B')


    source_distribution = source_distribution_6D(grid, Ti, mp)#, -3.*particles[0].vth)
    impurity_distribution = source_distribution_6D(grid, Ti, 10.81*mp)#, -3.*impurities[0].vth)
    num_deposited = 0
    num_sputtered = 0
    H_reflection_coefficient = 0.
    total_sputtering_yield = 0.
    run_tridyn = True

    #Construct energy and angle range and empty iead array
    angle_range = np.linspace(0.0, 90.0, num_angles)
    energy_range = np.linspace(0.1, 4.*phi_floating, num_energies)
    iead_average = np.zeros((len(energy_range), len(angle_range)))

    #Initialize figures
    plt.ion()
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    #Start of time loop
    time = 0.
    composition_B = 0.
    impurity_energies_out = []
    impurity_angles_out = []
    source_energies_out = []
    source_angles_out = []
    for time_index in range(timesteps+1):
        #Clear iead collection arrays
        energies_H = []
        angles_H = []
        energies_B = []
        angles_B = []

        #Clear plotting arrays
        positions = np.zeros(N)
        velocities = np.zeros(N)
        colors = np.zeros(N)

        time += dt
        grid.weight_particles_to_grid_boltzmann(particles, dt)
        print(f'n0: {grid.n0}\nadded_particles: {grid.added_particles}')
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        print(f'timestep: {time_index}')
        print(f'phi_max: {np.max(grid.phi)}')
        print(f'number deposited: {num_deposited}')
        print(f'number sputtered: {num_sputtered}')
        print(f'H reflection coefficient: {H_reflection_coefficient}')
        print(f'total B sputtering yield: {total_sputtering_yield}')
        print(f'active particles: {sum(1 for p in particles if p.is_active())}')

        #Begin particle loop
        for particle_index, particle in enumerate(particles):
            #If particle is active, push particles and store positions, velocities
            if particle.is_active():
                #Store particle coordinates for plotting
                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x

                if particle.charge_state == 0:
                    if particle.Z == 1:
                        colors[particle_index] = 0.0
                    if particle.Z == 5:
                        colors[particle_index] = 0.3
                elif particle.Z == 1:
                    colors[particle_index] = 0.6
                elif particle.Z == 5:
                    colors[particle_index] = 0.9

                #Check if particle is deactivated at interior domain
                source_particle_out = particle.apply_BCs_impurity_fluxes_out(grid, 1)
                if not source_particle_out is None:
                    source_energies_out.append(source_particle_out.kinetic_energy/e)
                    source_angles_out.append(source_particle_out.get_angle_wrt_wall())

                impurity_particle_out = particle.apply_BCs_impurity_fluxes_out(grid, 5)
                if not impurity_particle_out is None:
                    impurity_energies_out.append(impurity_particle_out.kinetic_energy/e)
                    impurity_angles_out.append(impurity_particle_out.get_angle_wrt_wall())

                #Interpolate E, push in time, and apply wall BCs
                particle.interpolate_electric_field(grid)
                particle.push_6D(dt)
                particle.apply_BCs_dirichlet(grid)

                #If particle is deactivated at wall, store in iead colleciton arrays
                if (not particle.is_active()) and particle.at_wall:
                    if particle.Z == 1:
                        energies_H.append(particle.kinetic_energy/e)
                        angles_H.append(particle.get_angle_wrt_wall())
                    #end if
                    if particle.Z == 5:
                        energies_B.append(particle.kinetic_energy/e)
                        angles_B.append(particle.get_angle_wrt_wall())
                    #end if

                if particle.Z == 5: composition_B += 1./len(particles)

                if particle.Z == 1 and particle.charge_state == 0 and particle.is_active():
                    particle.attempt_first_ionization(dt, cross_section, Te, grid)
                if particle.Z == 5 and particle.charge_state < 3 and particle.is_active():
                    particle.attempt_nth_ionization(dt, Te, grid)

            #If particle is not active, and domain needs more particles,
            # reinitialize as either source H
            else:
                #if np.random.choice((True, True), p=(0.90, 0.10)):
                if sum(1 for p in particles if (p.Z == 1 and p.is_active())) < source_N:
                    particle.reactivate(source_distribution, grid, time, p2c, 1.*mp, 1, 1)
                #else:
                #    particle.reactivate(impurity_distribution, grid, time, p2c, 10.81*mp, 1.*e, 5)
            #end if
        #end for particle_index, particle

        print(f'Percent Boron: {composition_B * 100.}')

        #Collect iead arrays into 2D IEAD histogram
        iead_H, energies_H_iead, angles_H_iead = np.histogram2d(energies_H, angles_H, bins=(num_energies,num_angles), range=((0., 400),(0., 90.)))
        iead += iead_H
        #iead_B, energies_B, angles_B = np.histogram2d(energies_B, angles_B, bins=(num_energies,num_angles))

        if run_tridyn:
            #Run F-TRIDYN for the collected IEADs
            #new_particle_list_H_s, new_particle_list_H_r = tridyn_interface.run_tridyn_simulations_from_iead(energies_H, angles_H, iead_H, number_histories=number_histories)
            #new_particle_list_B_s, new_particle_list_B_r = tridyn_interface_B.run_tridyn_simulations_from_iead(energies_B, angles_B, iead_B, number_histories=number_histories)

            new_particle_list_H_s, new_particle_list_H_r = tridyn_interface.run_tridyn_simulations_from_list(energies_H, angles_H, number_histories=number_histories)
            new_particle_list_B_s, new_particle_list_B_r = tridyn_interface_B.run_tridyn_simulations_from_list(energies_B, angles_B, number_histories=number_histories)

            try:
                total_sputtering_yield = (len(new_particle_list_H_s) + len(new_particle_list_B_s)) / (len(energies_H) + len(energies_B)) / number_histories
                H_reflection_coefficient = len(new_particle_list_H_r) / len(energies_H) / number_histories
            except ZeroDivisionError:
                print('WARNING: Divide by zero')
                total_sputtering_yield = 0
                H_reflection_coefficient = 0

            #Concatenate H and B lists from every NHth particle
            new_particle_list = new_particle_list_H_s[::number_histories] +\
                new_particle_list_H_r[::number_histories] +\
                new_particle_list_B_s[::number_histories] +\
                new_particle_list_B_r[::number_histories]

            #Count number of deposited Boron
            num_deposited += len(energies_B) - len(new_particle_list_B_r[::number_histories])

            num_sputtered += len(new_particle_list_B_s[::number_histories]) +\
                len(new_particle_list_H_s[::number_histories])

            #Create empty new particle array for reflected and sputtered particles
            new_particles = [None]*len(new_particle_list)
            composition_B = 0.
            for index, row in enumerate(new_particle_list):
                #Choose left or right wall, flip cos(alpha) appropriately
                if np.random.choice((True, False)):
                    x0 = np.random.uniform(0.0, 1.0)*grid.dx
                    row[1] = abs(row[1])
                else:
                    x0 = grid.length - np.random.uniform(0.0, 1.0)*grid.dx
                    row[1] = -abs(row[1])
                #end if
                #Create new particle
                new_particles[index] = particle_from_energy_angle_coordinates(*row, charge_state=0, p2c=p2c, T=Ti, grid=grid, x0=x0,
                    time=time, B=B)
                #Keep track of added charges for Botlzmann solver
                grid.add_particles(p2c)
            #end for

            #Concatenate particle and new particle lists
            particles += new_particles
            N = len(particles)
        #end if

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
            plt.scatter(positions, velocities, s=1, c=colors, cmap='winter')
            plt.axis([0.0, L, -400000, 400000])
            plt.draw()
            plt.savefig('pic_bca_ps'+str(time_index))
            plt.pause(0.001)

            plt.figure(3)
            plt.clf()
            plt.pcolormesh(angles_H_iead, energies_H_iead, iead)
            plt.draw()
            plt.pause(0.001)
        #end if
    #end for time_index

    #Create movies from .png plots
    c.convert('.', 'pic_bca_ps', 0,timesteps, 1, 'out_ps.gif')
    c.convert('.', 'pic_bca_phi', 0,timesteps, 1, 'out_phi.gif')

    breakpoint()
#end def pic_bca
