from pic.particle import *
from pic.grid import *
from pic.distributions import *
from pic.convert import *
from constants.constants import *
from fractal_tridyn.utils.generate_ftridyn_input import *

def pic_bca_aps():
    density = 5e16

    Ti = 10.*11600
    Te = 10.*11600

    wall = lithium
    source = hydrogen

    ng_per_debye = 2
    num_debye = 100

    nt_per_oscillation = 20
    n_ion_transit_times = 1

    ppc = 50

    psi = 0.0
    B0 = 1
    E0 = 0.

    deletion_step = 1
    plot_step = 10

    do_plots = True
    checkpoint_saving = 100
    do_checkpoint_saving = False
    checkpoint_load = 0
    load_checkpoint = False
    write_particles = True

    #TRIDYN params
    number_histories = 100

    #Physical parameters
    floating_potential = (Te/11600.)*0.5*np.log(1.*mp/2./np.pi/me/(1.+Ti/Te))
    print("floating potential value",floating_potential)

    LD = np.sqrt(kb*Te*epsilon0/e/e/density)
    ion_plasma_frequency = np.sqrt(density*e**2/mp/epsilon0)/2./np.pi

    omega=ion_plasma_frequency*2.0*np.pi
    #After you check if the code works on regular sheaths check how it works on RF sheaths
    RF_amptitude=10*Te
    #RF_amptitude=0*Te

    #Numerical parameters
    N = ppc*ng_per_debye*num_debye
    source_N = N
    L = num_debye*LD
    dt = 1./ion_plasma_frequency/nt_per_oscillation
    ng = num_debye*ng_per_debye
    p2c = density*L/N
    ion_thermal_velocity = np.sqrt(kb*Ti/mp)
    ion_transit_time = L/ion_thermal_velocity
    timesteps = int(n_ion_transit_times * ion_transit_time / dt)
    alpha = psi*np.pi/180.0

    B = np.array([B0*np.cos(alpha), B0*np.sin(alpha), 0.0])
    E = np.array([E0*np.cos(alpha), E0*np.sin(alpha), 0.0])

    if load_checkpoint:
        with open(f'particles{checkpoint_load}.sav', 'rb') as particle_file:
            particles = pickle.load(particle_file)
        with open(f'grid{checkpoint_load}.sav', 'rb') as grid_file:
            grid = pickle.load(grid_file)
        N = len(particles)
    else:
        grid = Grid(ng, L, Te, density,dt,omega,RF_amptitude,alpha, bc = 'dirichlet-dirichlet', tracked_ion_Z = 3)
        particles = [Particle(source['m'], 1, p2c, Ti, Z=source['Z'], B0=B, E0=E, grid=grid)
            for _ in range(N)]

    #tridyn_interface_source_wall = gen.tridyn_interface(source['symbol'], wall['symbol'])
    #tridyn_interface_wall_wall  = gen.tridyn_interface(wall['symbol'], wall['symbol'])

    tridyn_interface_source_wall = tridyn_interface(source['symbol'], wall['symbol'])
    tridyn_interface_wall_wall  = tridyn_interface(wall['symbol'], wall['symbol'])

    source_distribution = source_distribution_6D(grid, Ti, source['m'])

    color_dict = {
        (source['Z'], 1): 'red',
        (source['Z'], 0): 'maroon',
        (wall['Z'], 0): 'black',
        (wall['Z'], 1): 'blue',
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
        print(f'active  {wall["symbol"]} ions: {sum(1 for p in particles if p.is_active() and p.charge_state > 0 and p.Z==wall["Z"])}')

        active_source_particles.append(sum(1 for p in particles if p.is_active() and p.Z==source["Z"]))
        active_wall_particles.append(sum(1 for p in particles if p.is_active() and p.Z==wall["Z"]))

        #Final check before weighting - should only apply to BCA particles
        for particle in particles:
            particle.apply_BCs_dirichlet(grid)

        #Grid calculations
        grid.weight_particles_to_grid_boltzmann(particles)
        #grid.smooth_rho()
        grid.reference_density_update(time_index,"Elias")
        grid.reset_added_particles()
        grid.solve_for_phi_dirichlet_boltzmann()
        grid.differentiate_phi_to_E_dirichlet()

        #particle loop
        redeposited = 0
        heat_flux = 0.
        particle_flux = 0.
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

                if particle.Z == source['Z'] and particle.charge_state == 0 and particle.is_active():
                    particle.attempt_first_ionization(dt, Te, grid)

                if particle.Z == wall['Z'] and particle.charge_state < 3 and particle.is_active():
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
                    heat_flux += particle.kinetic_energy*particle.p2c/dt
                    particle_flux += particle.p2c

                #check if particle from wall leaves sheath region
                dx = grid.length/8
                if particle.from_wall and (grid.length/2 - dx < particle.x < grid.length/2 + dx):
                    particle.active = False
                    if particle.Z == source['Z']:
                        energies_out_source.append(particle.kinetic_energy/e)
                        angles_out_source.append(particle.get_angle_wrt_wall())
                        print(particle.v_x/particle.vth, file=f_source_out, flush=True)
                    if particle.Z == wall['Z']:
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

        print(f'heat flux: {heat_flux/1e6} MW/m2')
        print(f'particle flux: {particle_flux} 1/m2')

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
            new_particles[index].x += new_particles[index].v_x*dt*np.random.uniform(0, 1)

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
            plt.plot(np.linspace(0., grid.length, grid.ng), grid.n)
            plt.plot(np.linspace(0., grid.length, grid.ng), grid.n0*np.exp(e*(grid.phi - np.max(grid.phi))/kb/grid.Te))
            print(grid.n0, np.max(grid.n))
            #plt.axis([0.0, grid.length, 0.0, np.max(grid.rho)])
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
            for charge_state in range(0, grid.tracked_ion_Z + 1):
                plt.plot(np.linspace(0., grid.length, grid.ng), (grid.tracked_ion_density[charge_state] + grid.tracked_ion_density[charge_state][::-1])/2.)
            plt.axis([0.0, grid.length/2, 0.0, max([np.max(density) for density in grid.tracked_ion_density])])
            plt.draw()
            plt.savefig('pic_bca_tracked_ion_densities'+str(time_index))
            plt.pause(0.0001)

    #Create movies from .png plots
    convert('.', 'pic_bca_ps', 0,timesteps, plot_step, 'out_ps.gif')
    convert('.', 'pic_bca_phi', 0,timesteps, plot_step, 'out_phi.gif')

    print(np.sum(iead_out_source))
    print(np.sum(iead_out_wall))
    print(np.sum(iead_source))
    print(np.sum(iead_wall))

    #[iead_wall, iead_source, iead_out_wall, iead_out_source]

    breakpoint()
#end def pic_bca_aps
