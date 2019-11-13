from particle import *
from grid import *
from distributions import *
from convert import *
from constants import *

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

    tridyn_interface_D_B = gen.tridyn_interface('D', 'B')
    tridyn_interface_wall_wall = gen.tridyn_interface('B', 'B')

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

def pic_bca_aps():
    import fractal_tridyn.utils.generate_ftridyn_input as gen
    density = 1e19
    ng_per_LD = 3
    num_LD = 200
    ppc = 200
    dt = 8e-11
    timesteps = 10000

    Ti = 10.*11600
    Te = 50.*11600

    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    L = num_LD*LD
    ng = ng_per_LD*num_LD
    N = ng*ppc
    source_N = N
    p2c = density*L/N
    psi = 86.0
    alpha = psi*np.pi/180.0
    B0 = 2.
    E0 = 0.
    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])
    E = np.array([E0*np.cos(alpha), E0*np.sin(alpha), 0.0])

    deletion_step = 1
    plot_step = 20

    do_plots = True
    checkpoint_saving = 100
    do_checkpoint_saving = True
    checkpoint_load = 3900
    load_checkpoint = True
    write_particles = True

    #TRIDYN params
    number_histories = 100

    gold = {
        'Z': 79,
        'm': 196.967*mp,
        'symbol': 'Au'
    }

    argon = {
        'Z': 18,
        'm': 39.948*mp,
        'symbol': 'Ar'
    }

    helium = {
        'Z': 2,
        'm': 4*mp,
        'symbol': 'He'
    }

    boron = {
        'Z': 5,
        'm': 10.81*mp,
        'symbol': 'B'
    }

    hydrogen = {
        'Z': 1,
        'm': mp,
        'symbol': 'H'
    }

    wall = boron
    source = hydrogen

    if load_checkpoint:
        with open(f'particles{checkpoint_load}.sav', 'rb') as particle_file:
            particles = pickle.load(particle_file)
        with open(f'grid{checkpoint_load}.sav', 'rb') as grid_file:
            grid = pickle.load(grid_file)
        N = len(particles)
    else:
        grid = Grid(ng, L, Te, bc='dirichlet-dirichlet')
        particles = [Particle(source['m'], 1, p2c, Ti, Z=source['Z'], B0=B, E0=E, grid=grid)
            for _ in range(N)]

    tridyn_interface_source_wall = gen.tridyn_interface(source['symbol'], wall['symbol'])
    tridyn_interface_wall_wall  = gen.tridyn_interface(wall['symbol'], wall['symbol'])

    source_distribution = source_distribution_6D(grid, Ti, source['m'])

    color_dict = {
        (source['Z'], 1): 'red',
        (wall['Z'], 1): 'blue',
        (source['Z'], 0): 'maroon',
        (wall['Z'], 0): 'black',
        (wall['Z'], 2): 'green',
        (wall['Z'], 3): 'purple'
    }

    size_dict = {
        1: 0.5,
        0.: 1.0
    }

    active_source_particles = []
    active_wall_particles = []

    wall_particles_sputtered = []
    wall_particles_self_sputtered = []
    wall_particles_source_sputtered = []
    wall_particles_incident = []
    source_particles_incident = []
    source_particles_reflected = []
    wall_particles_reflected = []
    redeposition_fraction = []
    deletion_flags = []
    time = 0.

    energies_iead = np.linspace(0.0, 300.0, 41)
    angles_iead = np.linspace(0.0, 90.0, 31)
    iead_source = np.zeros((len(energies_iead)-1, len(angles_iead)-1))
    iead_wall = np.zeros((len(energies_iead)-1, len(angles_iead)-1))
    iead_out_source = np.zeros((len(energies_iead)-1, len(angles_iead)-1))
    iead_out_wall = np.zeros((len(energies_iead)-1, len(angles_iead)-1))

    if do_plots:
        plt.ion()
        plt.figure(1)
        plt.figure(2)
        plt.figure(3)
        plt.figure(4)
        plt.figure(5)
        plt.figure(6)
        plt.figure(7)
        plt.figure(8)

    if write_particles:
        f_source_wall = open('source_wall.dat', 'a')
        f_wall_wall = open('wall_wall.dat', 'a')
        f_source_from_wall = open('source_from_wall.dat', 'a')
        f_wall_from_wall = open('wall_from_wall.dat', 'a')
        f_source_out = open('source_out.dat', 'a')
        f_wall_out = open('wall_out.dat', 'a')
        f_lateral_displacements = open('lateral_displacements.dat', 'a')

    for time_index in range(timesteps):
        energies_source = []
        angles_source = []
        energies_wall = []
        angles_wall = []
        energies_out_source = []
        energies_out_wall = []
        angles_out_source = []
        angles_out_wall = []
        lateral_displacements_boron = []

        positions = np.zeros(N)
        velocities = np.zeros(N)
        colors = ['black']*N
        sizes = [1.0]*N

        time += dt
        print(f'time: {time_index}')

        try:
            print(f'total yield: {sum(wall_particles_sputtered) / sum(wall_particles_incident + source_particles_incident)}')
            print(f'{source["symbol"]} reflection: {sum(source_particles_reflected) / sum(source_particles_incident)}')
        except ZeroDivisionError:
            pass
        try:
            print(f'{wall["symbol"]} reflection: {sum(wall_particles_reflected) / sum(wall_particles_incident)}')
            print(f'{wall["symbol"]} self sputtering yield: {sum(wall_particles_self_sputtered) / sum(wall_particles_incident)}')
        except ZeroDivisionError:
            pass
        print(f'active particles: {sum(1 for p in particles if p.is_active())}')
        print(f'active {source["symbol"]} particles: {sum(1 for p in particles if p.is_active() and p.Z==source["Z"])}')
        print(f'active  {wall["symbol"]} particles: {sum(1 for p in particles if p.is_active() and p.Z==wall["Z"])}')

        active_source_particles.append(sum(1 for p in particles if p.is_active() and p.Z==source["Z"]))
        active_wall_particles.append(sum(1 for p in particles if p.is_active() and p.Z==wall["Z"]))

        #Final check before weighting - should only apply to BCA particles
        for particle in particles:
            particle.apply_BCs_dirichlet(grid)

        #Grid calculations
        grid.weight_particles_to_grid_boltzmann(particles, dt)
        grid.smooth_rho()
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        #particle loop
        redeposited = 0
        for particle_index, particle in enumerate(particles):
            if particle.is_active():
                particle.interpolate_electric_field(grid)
                particle.push_6D(dt)
                particle.apply_BCs_dirichlet(grid)

                #plotting storage
                positions[particle_index] = particle.x
                velocities[particle_index] = particle.v_x/particle.vth
                colors[particle_index] = color_dict[particle.Z, particle.charge_state]
                sizes[particle_index] = size_dict[particle.charge_state]

                if particle.Z == 1 and particle.charge_state == 0 and particle.is_active():
                    particle.attempt_first_ionization(dt, Te, grid)
                if particle.Z == 5 and particle.charge_state < 3 and particle.is_active():
                    particle.attempt_nth_ionization(dt, Te, grid)

                #particle just deactivated at wall
                if not particle.is_active() and particle.at_wall:
                    if particle.Z == wall['Z']:
                        energies_wall.append(particle.kinetic_energy/e)
                        angles_wall.append(particle.get_angle_wrt_wall())
                        print(particle.v_x/particle.vth, file=f_wall_wall, flush=True)
                        print(np.sqrt(particle.y**2 + particle.z**2), file=f_lateral_displacements, flush=True)
                        if particle.from_wall:
                            redeposited += 1
                    if particle.Z == source['Z']:
                        energies_source.append(particle.kinetic_energy/e)
                        angles_source.append(particle.get_angle_wrt_wall())
                        print(particle.v_x/particle.vth, file=f_source_wall, flush=True)

                #check if particle from wall leaves sheath region
                dx = grid.length/8
                if particle.from_wall and (grid.length/2 - dx < particle.x < grid.length/2 + dx):
                    particle.active = False
                    if particle.Z == source['Z']:
                        energies_out_source.append(particle.kinetic_energy/e)
                        angles_out_source.append(particle.get_angle_wrt_wall())
                        print(particle.v_x/particle.vth, file=f_source_out, flush=True)
                    if particle.Z == wall['Z']:
                        print('Boron left!')
                        energies_out_wall.append(particle.kinetic_energy/e)
                        angles_out_wall.append(particle.get_angle_wrt_wall())
                        print(particle.v_x/particle.vth, file=f_wall_out, flush=True)

            else: #particle is not active at start of loop - redistribute as source gas to maintain constant ion density
                if sum(1 for p in particles if (p.Z==source['Z'] and p.is_active() and p.charge_state>0)) < source_N:
                    particle.reactivate(source_distribution, grid, time, p2c, source['m'], 1, source['Z'])
                    particle.from_wall = 0
                    particle.at_wall = 0
                else:
                    deletion_flags.append(particle_index)
        #end particle loop

        #Run deletion loop
        if time_index%deletion_step == 0:
            new_particles_length = len(particles) - len(deletion_flags)
            new_particles = [None]*new_particles_length
            counter = 0
            for particle_index, particle in enumerate(particles):
                if not particle_index in deletion_flags:
                    new_particles[counter] = particle
                    counter += 1
            particles = new_particles
            print(f'{len(deletion_flags)} particles deleted!')
            deletion_flags = []

        print(f'average {source["symbol"]} energy: {np.mean(energies_source)}')
        print(f'average {source["symbol"]} angle: {np.mean(angles_source)}')

        print(f'average {wall["symbol"]} energy: {np.mean(energies_wall)}')
        print(f'average {wall["symbol"]} angle: {np.mean(angles_wall)}')

        sputtered_source_wall, reflected_source_wall = tridyn_interface_source_wall.run_tridyn_simulations_from_list(energies_source, angles_source, number_histories=number_histories)
        sputtered_wall_wall, reflected_wall_wall = tridyn_interface_wall_wall.run_tridyn_simulations_from_list(energies_wall, angles_wall, number_histories=number_histories)

        bins = (energies_iead, angles_iead)
        bins_wall = (energies_iead, angles_iead)
        iead_source_tmp = np.histogram2d(energies_source, angles_source, bins)
        iead_wall_tmp = np.histogram2d(energies_wall, angles_wall, bins_wall)
        iead_out_source_tmp = np.histogram2d(energies_out_source, angles_out_source, bins)
        iead_out_wall_tmp = np.histogram2d(energies_out_wall, angles_out_wall, bins_wall)

        iead_source += iead_source_tmp[0]
        iead_wall += iead_wall_tmp[0]
        iead_out_source += iead_out_source_tmp[0]
        iead_out_wall += iead_out_wall_tmp[0]

        wall_particles_incident.append(len(energies_wall))
        source_particles_incident.append(len(energies_source))

        wall_particles_sputtered.append(len(sputtered_source_wall[::number_histories]) + len(sputtered_wall_wall[::number_histories]))
        wall_particles_self_sputtered.append(len(sputtered_wall_wall[::number_histories]))
        wall_particles_source_sputtered.append(len(sputtered_source_wall[::number_histories]))
        source_particles_reflected.append(len(reflected_source_wall[::number_histories]))
        wall_particles_reflected.append(len(reflected_wall_wall[::number_histories]))
        try:
            redeposition_fraction.append(1 - len(reflected_wall_wall[::number_histories])/len(energies_wall))
        except ZeroDivisionError:
            redeposition_fraction.append(0)

        new_particle_list = sputtered_source_wall[::number_histories] + \
            reflected_source_wall[::number_histories] + \
            sputtered_wall_wall[::number_histories] + \
            reflected_wall_wall[::number_histories]

        new_particles = [None]*len(new_particle_list)
        for index, row in enumerate(new_particle_list):

            if np.random.choice((True, False)):
                x0 = 0.0
                row[1] = abs(row[1])
            else:
                x0 = grid.length
                row[1] = -abs(row[1])

            new_particles[index] = particle_from_energy_angle_coordinates(*row, dt,
                charge_state=0, p2c=p2c, T=Ti, grid=grid, x0=x0, time=time, B=B)
            #grid.add_particles(p2c)
            if new_particles[index].Z == wall['Z']:
                print(new_particles[index].v_x/new_particles[index].vth, file=f_wall_from_wall, flush=True)
            if new_particles[index].Z == source['Z']:
                print(new_particles[index].v_x/new_particles[index].vth, file=f_source_from_wall, flush=True)

        sputtered_energies = [particle.kinetic_energy/e for particle in new_particles]

        particles += new_particles
        N = len(particles)

        if time_index%checkpoint_saving==0 and do_checkpoint_saving:
            with open(f'particles{time_index}.sav', 'wb') as particle_file:
                pickle.dump(particles, particle_file)
            with open(f'grid{time_index}.sav', 'wb') as grid_file:
                pickle.dump(grid, grid_file)
            print('Checkpoint saved!')

        if time_index%plot_step==0 and do_plots:
            plt.figure(1)
            plt.clf()

            velocities = [velocity if position < grid.length/2 else -velocity for velocity, position in zip(velocities, positions)]
            positions = [position if position < grid.length/2 else grid.length - position for position in positions]

            plt.scatter(positions, velocities, s=sizes, c=colors)
            plt.axis([0.0, grid.length/4, -8., 8.])
            plt.draw()
            plt.savefig('pic_bca_ps'+str(time_index))
            plt.pause(0.0001)

            plt.figure(2)
            plt.clf()
            plt.plot(np.linspace(0., grid.length, grid.ng), grid.phi)
            plt.axis([0.0, grid.length, 0.0, np.max(grid.phi)])
            plt.draw()
            plt.savefig('pic_bca_phi'+str(time_index))
            plt.pause(0.0001)

            plt.figure(3)
            plt.clf()
            plt.plot(np.linspace(0., grid.length, grid.ng), grid.rho)
            plt.axis([0.0, grid.length, 0.0, np.max(grid.rho)])
            plt.draw()
            plt.savefig('pic_bca_rho'+str(time_index))
            plt.pause(0.0001)

            plt.figure(4)
            plt.clf()
            plt.pcolormesh(angles_iead[:-1], energies_iead[:-1], iead_source)
            #plt.contourf(angles_iead[:-1], energies_iead[:-1], iead_source)
            plt.title(f'{source["symbol"]} IEAD ')
            plt.draw()
            plt.savefig('pic_bca_iead_He')
            plt.pause(0.0001)

            plt.figure(5)
            plt.clf()
            plt.pcolormesh(angles_iead[:-1], energies_iead[:-1], iead_wall)
            plt.title(f'{wall["symbol"]} IEAD ')
            plt.draw()
            plt.savefig('pic_bca_iead_Au')
            plt.pause(0.0001)

            plt.figure(6)
            plt.clf()
            plt.pcolormesh(angles_iead[:-1], energies_iead[:-1], iead_out_source)
            plt.title(f'{source["symbol"]} Flux Out ')
            plt.draw()
            plt.savefig('pic_bca_out_He')
            plt.pause(0.0001)

            plt.figure(7)
            plt.clf()
            plt.pcolormesh(angles_iead[:-1], energies_iead[:-1], iead_out_wall)
            plt.title(f'{wall["symbol"]} Flux Out ')
            plt.draw()
            plt.savefig('pic_bca_out_Au')
            plt.pause(0.0001)

            plt.figure(8)
            plt.clf()
            plt.scatter(positions, velocities, s=sizes, c=colors)
            plt.axis([0.0, grid.length/8, -6, 6.])
            plt.draw()
            plt.savefig('pic_bca_ps_zoomed'+str(time_index))
            plt.pause(0.0001)

    #Create movies from .png plots
    c.convert('.', 'pic_bca_ps', 0,timesteps, plot_step, 'out_ps.gif')
    c.convert('.', 'pic_bca_phi', 0,timesteps, plot_step, 'out_phi.gif')

    print(np.sum(iead_out_source))
    print(np.sum(iead_out_wall))
    print(np.sum(iead_source))
    print(np.sum(iead_wall))

    #[iead_wall, iead_source, iead_out_wall, iead_out_source]

    breakpoint()

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
    tridyn_interface = gen.tridyn_interface('H', 'B')
    tridyn_interface_B = gen.tridyn_interface('B', 'B')
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
