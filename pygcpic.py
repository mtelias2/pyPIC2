import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import convert as c
import itertools
import pickle

epsilon0 = 8.854e-12
e = 1.602e-19
mp = 1.67e-27
me = 9.11e-31
kb = 1.38e-23

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

def particle_from_energy_angle_coordinates(energy, ca, cb, cg, m, Z, dt,
    B=None, charge_state=0, p2c=0, T=0., grid=None, x0=0., time=0.):
    '''
    This function creates and initializes a Particle object using energy-angle
    coordintes (e.g., those from F-TRIDYN output).

    Args:
        energy (float): particle kinetic energy
        ca (float): directional cosine along x-axis, range 0. to 1.
        cb (float): directional cosine along y-axis, range 0. to 1.
        cg (float): directional cosine along z-axis, range 0. to 1.
        m (float): particle mass in kg
        Z (int): particle atomic number
        B (ndarray), optional: magnetic field (assumed zero)
        charge_state (int), optional: particle charge state (assumed 0)
        p2c (int), optional: assumed zero (i.e., chargeless tracer)
        T (float), optional: species temperature (assumed zero)
        grid (Grid), optional: grid associated with particle, assumed
            None
        x0 (float), optional: starting position along x-axis (assumed zero)
        time (float), optional: particle's current time (assumed zero)
    '''
    speed = np.sqrt(2.*energy*e/(m*mp))
    u = [ca, cb, cg]

    u /= np.linalg.norm(u)
    v = speed * u
    particle = Particle(m*mp, charge_state, p2c, T, Z, grid=grid)
    particle.r[3:6] = v
    particle.r[0] = x0 + np.random.uniform(0.0, 1.0)*v[0]*dt
    particle.time = time
    particle.B[:] = B
    particle.from_wall = 1
    return particle
#end def particle_from_energy_angle_coordinates

class Particle:
    '''
        Generic particle object. Can work in 6D or GC coordinate systems and
        can transform between the two representations on the fly. Includes
        methods for changing the particle's properties and state, and for
        advancing the particle forward in time in either coordinate system.
    '''
    def __init__(self, m, charge_state, p2c, T, Z, B0=np.zeros(3), E0=np.zeros(3),
        grid=None, vx=0.):
        '''
        Particle initialization.

        Args:
            m (float): mass in kg
            charge_state (int): charge state
            p2c (float): number of physical particles represented by this
                particle. Should be > 1 except for tracers when p2c = 0.
            T (float): species temperature in K
            Z (int): species atomic number
            B0 (ndarray): magnetic field vector (assumed zero)
            E0 (ndarray): electric field vector (assumed zero)
            grid (Grid), optional: grid object associated with this
                particle (assumed None)
        '''
        self.r = np.zeros(7)
        self.charge_state = charge_state
        self.Z = Z
        self.m = m
        self.T = T
        self.p2c = p2c
        self.vth = np.sqrt(kb*self.T/self.m)
        self.mode = 0
        #6D mode: 0
        #GC mode: 1
        self.E = E0
        self.B = B0

        #Electric field at particle position
        self.active = 1
        self.at_wall = 0
        self.from_wall = 0
        if grid != None: self._initialize_6D(grid, vx=vx)
    #end def __init__

    def __repr__(self):
        return f'Particle({self.m}, {self.charge_state}, {self.p2c}, {self.T}, {self.Z})'
    #end def

    def is_active(self):
        '''
        Returns a boolean that is true if the particle is active and false
        if the particle is inactive.

        Returns:
            is_active (bool): whether the particle is active or not
        '''
        return self.active == 1
    #end def is_active

    @property
    def speed(self):
        '''
        Returns the particle's total speed.

        Tests:

        >>> particle = Particle(1.0, 1/e, 1.0, 1.0, 1)
        >>> particle.r[3] = 1.0
        >>> particle.r[4:6] = 2.0
        >>> particle.speed
        3.0
        '''
        return np.sqrt(self.r[3]**2 + self.r[4]**2 + self.r[5]**2)
    #end def speed

    @speed.setter
    def speed(self, speed):
        '''
        Scales the particle's speed to the given speed retaining direction.

        Args:
            speed (float): new speed to scale to.

        Tests:
            >>> particle = Particle(1.0, 1/e, 1.0, 1.0, 1)
            >>> particle.r[3] = 1.0
            >>> particle.speed = 2.0
            >>> particle.speed
            2.0
        '''
        u = self.v / np.linalg.norm(self.v)
        self.v = u*speed
    #end def speed

    @property
    def x(self):
        '''
        Returns the particle's x position.

        Returns:
            x (float): x position
        '''
        return self.r[0]
    #end def x

    @property
    def y(self):
        return self.r[1]
    #end def y

    @property
    def z(self):
        return self.r[2]
    #end def z

    @x.setter
    def x(self, x0):
        '''
        Allows the setting of r[0] with the .x accsessor

        Notes:
            Can be used in either GC or 6D mode.

        Tests:
            >>> particle = Particle(1.0, 1/e, 1.0, 1.0, 1)
            >>> particle.x = 10.0
            >>> particle.r[0]
            10.0
        '''
        self.r[0] = x0
    #end def x

    @property
    def v_x(self):
        '''
        Returns the particle's x-velocity.

        Returns:
            v_x (float): x velocity
        '''
        return self.r[3]
    #end def v_x

    @v_x.setter
    def v_x(self, v_x):
        self.r[3] = v_x
    #end def v_x

    @property
    def v(self):
        return self.r[3:6]
    #end def v

    @v.setter
    def v(self, v0):
        self.r[3:6] = v0
    #end def

    def get_angle_wrt_wall(self, use_degrees=True):
        '''
        Returns the particle's angle with respect to the normal of the y-x
        plane in degrees. Default return value is in degrees for F-Tridyn
        input.

        Args:
            use_degrees (bool), optional: Whether to use degrees (as opposed
            to radians) for the return value.

        Returns:
            alpha (float): angle w.r.t. y-x plane wall.

        Tests:

        >>> np.random.seed(1)
        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> particle.r[3] = np.random.uniform(0.0, 1.0)
        >>> particle.get_angle_wrt_wall(use_degrees=True)
        0.0
        >>> particle.get_angle_wrt_wall(use_degrees=False)
        0.0
        '''
        v = self.r[3:6]
        vyz = np.sqrt(v[1]**2 + v[2]**2)
        alpha = np.arctan2(vyz, np.abs(v[0]))
        if use_degrees:
            return alpha*180./np.pi
        else:
            return alpha
        #end if
    #end def get_angle_wrt_wall

    @property
    def kinetic_energy(self):
        '''
        Returns the particle's kinetic energy.

        Tests:

        >>> particle=Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> particle.r[3] = 1.0
        >>> particle.r[4:6] = 2.0
        >>> particle.kinetic_energy
        4.5
        '''
        return 0.5*self.m*self.speed**2
    #end def kinetic_energy

    def _initialize_6D(self, grid, vx=0.):
        '''
        Given a grid object, initialize the particle on the grid with a
        uniform distribution in space and a normal distribution of speeds
        based on its thermal velocity.

        Args:
            grid (Grid): the grid with which the particle is
                associated

        Tests:

        >>> np.random.seed(1)
        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> grid = Grid(100, 1.0, 1.0)
        >>> particle._initialize_6D(grid)
        >>> np.random.seed(1)
        >>> particle.r[0] == np.random.uniform(0.0, grid.length)
        True
        >>> particle.r[3] == np.random.normal(0.0, particle.vth, 3)[0]
        True
        '''
        self.r[0] = np.random.uniform(0.0, grid.length)
        self.r[1:3] = 0.0
        self.r[3:6] = np.random.normal(0.0, self.vth , 3) + vx
        self.r[3] = self.r[3]
        self.r[6] = 0.0
    #end def initialize_6D

    def set_x_direction(self, direction):
        '''
        Set the direction of the particle by taking the absolute value of its
        x-velocity and, if necessary, negating it.

        Args:
            direction (str): 'left' or 'right'
        '''
        if direction.lower() == 'left':
            self.r[3] = -abs(self.r[3])
        elif direction.lower() == 'right':
            self.r[3] = abs(self.r[3])
        elif type(direction) == type(''):
            raise ValueError('particle.set_x_direction() received neither right nor left')
        else:
            raise TypeError('particle.set_x_direction(direction) received a non-string type for direction')
        #end if
    #end def set_x_direction

    def interpolate_electric_field_dirichlet(self, grid):
        '''
        Interpolates electric field values from grid to particle position
        assuming Dirichlet-Dirichlet boundary conditions.

        Args:
            grid (Grid): the grid with which the particle is
                associated

        Tests:

        >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
        >>> grid = Grid(100, 1.0, 1.0)
        >>> particle._initialize_6D(grid)
        >>> grid.E[:] = 1.0
        >>> particle.interpolate_electric_field_dirichlet(grid)
        >>> particle.E[0]
        1.0
        '''
        ind = int(np.floor(self.x/grid.dx))
        w_l = (self.x%grid.dx)/grid.dx
        w_r = 1.0 - w_l
        self.E[0] = grid.E[ind]*w_l + grid.E[ind+1]*w_r
    #end def interpolate_electric_field

    def attempt_first_ionization(self, dt, temperature, grid):
        '''
        Monte Carlo Collision of first ionization, assuming constant cross-section.

        Args:
            dt (float): timestep in s
            cross_section (float): cross section in m2
            temperature (float): background temperature in K
            density (float): background density in m-3

        '''
        if self.Z == 5:
            Te = [8.626E-01, 1.329E+00, 2.160E+00, 3.140E+00, 4.314E+00, 5.741E+00,
            7.508E+00, 9.746E+00, 1.267E+01, 1.660E+01, 2.212E+01, 3.034E+01,
            4.353E+01, 6.704E+01, 1.162E+02, 2.490E+02, 8.265E+02, 8.481E+03,
            8.669E+04]

            R_cm3_s = [1.057E-12, 3.996E-11, 5.912E-10, 2.458E-09, 6.083E-09, 1.155E-08,
            1.878E-08, 2.767E-08, 3.806E-08, 4.979E-08, 6.257E-08, 7.590E-08,
            8.901E-08, 1.005E-07, 1.080E-07, 1.079E-07, 9.470E-08, 5.161E-08,
            2.159E-08]
        elif self.Z == 1:
             Te = [8.626E-01, 1.011E+00, 2.178E+00, 3.539E+00, 5.146E+00, 7.069E+00,
             9.410E+00, 1.231E+01, 1.598E+01, 2.076E+01, 2.720E+01, 3.625E+01,
             4.973E+01, 7.133E+01, 1.099E+02, 1.904E+02, 4.079E+02, 1.355E+03,
             1.390E+04, 8.595E+04]

             R_cm3_s = [7.553E-16, 8.291E-15, 1.714E-11, 2.470E-10, 9.985E-10, 2.398E-09,
             4.412E-09, 6.940E-09, 9.869E-09, 1.309E-08, 1.649E-08, 1.996E-08,
             2.329E-08, 2.624E-08, 2.834E-08, 2.881E-08, 2.627E-08, 1.926E-08,
             8.109E-09, 3.829E-09]

        Te_K = [T*11600. for T in Te]
        R_m3_s = [R/1e6 for R in R_cm3_s]

        ionization_rate = np.interp(temperature, Te_K, R_m3_s)

        index_l = int(np.floor(self.x/grid.dx))
        index_r = (index_l + 1)
        w_r = (self.x%grid.dx)/grid.dx
        w_l = 1.0 - w_r
        density = w_l*grid.n[index_l] + w_r*grid.n[index_r]
        probability = density**2 * ionization_rate * grid.dx * dt / self.p2c

        #print(f'p: {probability} mfp: {mfp} n: {density} speed: {self.speed}')

        if np.random.uniform(0., 1.) < probability and self.charge_state == 0.:
            self.charge_state = 1
            grid.add_particles(self.p2c)

    def attempt_nth_ionization(self, dt, temperature, grid):
        '''
        Monte Carlo Collision of nth ionization, assuming constant cross-section.

        Args:
            dt (float): timestep in s
            cross_section (float): cross section in m2
            temperature (float): background temperature in K
            density (float): background density in m-3

        '''
        if self.Z == 5:
            if self.charge_state == 0:
                Te = [8.626E-01, 1.329E+00, 2.160E+00, 3.140E+00, 4.314E+00, 5.741E+00,
                7.508E+00, 9.746E+00, 1.267E+01, 1.660E+01, 2.212E+01, 3.034E+01,
                4.353E+01, 6.704E+01, 1.162E+02, 2.490E+02, 8.265E+02, 8.481E+03,
                8.669E+04]

                R_cm3_s = [1.057E-12, 3.996E-11, 5.912E-10, 2.458E-09, 6.083E-09, 1.155E-08,
                1.878E-08, 2.767E-08, 3.806E-08, 4.979E-08, 6.257E-08, 7.590E-08,
                8.901E-08, 1.005E-07, 1.080E-07, 1.079E-07, 9.470E-08, 5.161E-08,
                2.159E-08]
            elif self.charge_state == 1:
                Te = [8.612E-01, 1.869E+00, 4.028E+00, 6.547E+00, 9.522E+00, 1.308E+01,
                1.741E+01, 2.276E+01, 2.956E+01, 3.840E+01, 5.031E+01, 6.707E+01,
                9.203E+01, 1.319E+02, 2.033E+02, 3.522E+02, 7.547E+02, 2.505E+03,
                2.571E+04, 8.582E+04]
                R_cm3_s = [1.375E-21, 1.396E-14, 2.693E-11, 3.643E-10, 1.393E-09, 3.188E-09,
                5.629E-09, 8.554E-09, 1.182E-08, 1.533E-08, 1.900E-08, 2.273E-08,
                2.639E-08, 2.972E-08, 3.221E-08, 3.300E-08, 3.032E-08, 2.252E-08,
                9.306E-09, 5.538E-09]
            elif self.charge_state == 2:
                Te = [1.366E+00, 2.819E+00, 6.073E+00, 9.875E+00, 1.436E+01, 1.972E+01,
                2.624E+01, 3.432E+01, 4.456E+01, 5.790E+01, 7.587E+01, 1.012E+02,
                1.387E+02, 1.990E+02, 3.064E+02, 5.311E+02, 1.138E+03, 3.778E+03,
                3.877E+04, 8.602E+04]
                R_cm3_s = [1.230E-21, 2.871E-15, 5.524E-12, 7.439E-11, 2.824E-10, 6.401E-10,
                1.117E-09, 1.677E-09, 2.293E-09, 2.946E-09, 3.629E-09, 4.337E-09,
                5.055E-09, 5.759E-09, 6.382E-09, 6.779E-09, 6.575E-09, 5.269E-09,
                2.483E-09, 1.829E-09]

        Te_K = [T*11600. for T in Te]
        R_m3_s = [R/1e6 for R in R_cm3_s]

        ionization_rate = np.interp(temperature, Te_K, R_m3_s)

        index_l = int(np.floor(self.x/grid.dx))
        index_r = (index_l + 1)
        w_r = (self.x%grid.dx)/grid.dx
        w_l = 1.0 - w_r
        density = w_l*grid.n[index_l] + w_r*grid.n[index_r]
        probability = density**2 * ionization_rate * grid.dx * dt / self.p2c

        #print(f'p: {probability} mfp: {mfp} n: {density} speed: {self.speed}')

        if np.random.uniform(0., 1.) < probability and self.charge_state == 0.:
            print(f'Ionized boron from {self.charge_state} to {self.charge_state+1}!')
            self.charge_state += 1
            grid.add_particles(self.p2c)

    def push_6D(self,dt):
        '''
        Boris-Buneman integrator that pushes the particle in 6D cooordinates
        one timeste of magnitude dt.

        Args:
            dt (float): timestep

        Tests:
            >>> particle = Particle(1.0, 1/e, 1.0, 1.0, 1)
            >>> grid = Grid(100, 1.0, 1.0)
            >>> particle._initialize_6D(grid)
            >>> particle.r[3:6] = 0.0
            >>> grid.E[0] = 1.0
            >>> particle.push_6D(1.0)
            >>> particle.r[3]
            1.0
        '''
        constant = 0.5*dt*self.charge_state*1.602e-19/self.m

        self.r[3] += constant*self.E[0]

        tx = constant*self.B[0]
        ty = constant*self.B[1]
        tz = constant*self.B[2]

        t2 = tx*tx + ty*ty + tz*tz

        sx = 2.*tx / (1. + t2)
        sy = 2.*ty / (1. + t2)
        sz = 2.*tz / (1. + t2)

        vfx = self.r[3] + self.r[4]*tz - self.r[5]*ty
        vfy = self.r[4] + self.r[5]*tx - self.r[3]*tz
        vfz = self.r[5] + self.r[3]*ty - self.r[4]*tx

        self.r[3] += vfy*sz - vfz*sy
        self.r[4] += vfz*sx - vfx*sz
        self.r[5] += vfx*sy - vfy*sx

        self.r[3] += constant*self.E[0]

        self.r[0] += self.r[3]*dt
        self.r[1] += self.r[4]*dt
        self.r[2] += self.r[5]*dt

        self.r[6] += dt
    #end push_6D

    def transform_6D_to_GC(self):
        '''
        Transform the particle state vector from 6D to guiding-center
        coordinates. This process results in the loss of one coordinate
        which represents the phase of the particle.

        Tests:
            Tests that vpar and total speed are conserved in transforming.
            >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
            >>> particle.B[:] = np.random.uniform(0.0, 1.0, 3)
            >>> grid = Grid(100, 1.0, 1.0e9)
            >>> v_x = particle.r[3]
            >>> speed = particle.speed
            >>> particle._initialize_6D(grid)
            >>> particle.transform_6D_to_GC()
            >>> particle.transform_GC_to_6D()
            >>> round(v_x,6) == round(particle.r[3],6)
            True
            >>> round(speed,6) == round(particle.speed,6)
            True
        '''
        x = self.r[0:3]
        v = self.r[3:6]
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2
        b = self.B/np.sqrt(B2)

        vpar_mag = v.dot(b)
        vpar = vpar_mag*b
        wc = abs(self.charge_state)*e*np.sqrt(B2)/self.m
        rho = vpar_mag/wc
        vperp = v - vpar
        vperp_mag = np.sqrt(vperp[0]**2 + vperp[1]**2 + vperp[2]**2)
        vperp_hat = vperp/vperp_mag
        mu = 0.5*self.m*vperp_mag**2/np.sqrt(B2)
        rl_mag = vperp_mag/wc
        rl_hat = -np.sign(self.charge_state)*e*np.cross(vperp_hat,b)
        rl = rl_mag*rl_hat

        self.r[0:3] = x - rl
        self.r[3] = vpar_mag
        self.r[4] = mu
        self.mode = 1
    #end def transform_6D_to_GC

    def transform_GC_to_6D(self):
        '''
        Transform the particle state vector from guiding-center to 6D
        coordinates. This method uses a single random number to generate the
        missing phase information from the GC coordinates.

        Tests:
            Tests that vpar and total speed are conserved in transforming.
            >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
            >>> particle.B[:] = np.random.uniform(0.0, 1.0, 3)
            >>> grid = Grid(100, 1.0, 1.0e9)
            >>> v_x = particle.r[3]
            >>> speed = particle.speed
            >>> particle._initialize_6D(grid)
            >>> particle.transform_6D_to_GC()
            >>> particle.transform_GC_to_6D()
            >>> round(v_x,6) == round(particle.r[3],6)
            True
            >>> round(speed,6) == round(particle.speed,6)
            True
        '''
        X = self.r[0:3]
        vpar_mag = self.r[3]
        mu = self.r[4]
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2
        b = self.B/np.sqrt(B2)

        vperp_mag = np.sqrt(2.0*mu*np.sqrt(B2)/self.m)
        wc = abs(self.charge_state)*e*np.sqrt(B2)/self.m
        rl_mag = vperp_mag/wc
        a = np.random.uniform(0.0, 1.0, 3)
        aperp = a - a.dot(b)
        aperp_mag = np.sqrt(aperp[0]**2 + aperp[1]**2 + aperp[2]**2)
        bperp_hat = aperp/aperp_mag
        rl = rl_mag*bperp_hat
        x = X + rl
        vperp_hat = np.cross(b, bperp_hat)
        v = vpar_mag*b + vperp_mag*vperp_hat

        self.r[0:3] = x
        self.r[3:6] = v
        self.r[6] = self.r[6]
        self.mode = 0
    #end def transform_GC_to_6D

    def push_GC(self,dt):
        '''
        Push the particle using the guiding-center cooordinates one timestep
        of magnitude dt.

        Args:
            dt (float): timestep
        '''
        #Assuming direct time-independence of rdot
        r0 = self.r
        k1 = dt*self._eom_GC(r0)
        k2 = dt*self._eom_GC(r0 + k1/2.)
        k3 = dt*self._eom_GC(r0 + k2/2.)
        k4 = dt*self._eom_GC(r0 + k3)
        self.r += (k1 + 2.*k2 + 2.*k3 + k4)/6.
        self.r[6] += dt
    #end def push_GC

    def _eom_GC(self,r):
        '''
        An internal method that calculates the differential of the r-vector
        for the equation of motion given to the RK4 guiding-center solver.

        Args:
            r (ndarray): particle state vector in GC coordinates
        '''
        B2 = self.B[0]**2 + self.B[1]**2 + self.B[2]**2

        b0 = self.B[0]/np.sqrt(B2)
        b1 = self.B[1]/np.sqrt(B2)
        b2 = self.B[2]/np.sqrt(B2)

        wc = abs(self.charge_state)*e*np.sqrt(B2)/self.m
        rho = r[3]/wc

        rdot = np.zeros(7)

        rdot[0] = (self.E[1]*self.B[2] - self.E[2]*self.B[1])/B2 + r[3]*b0
        rdot[1] = (self.E[2]*self.B[0] - self.E[0]*self.B[2])/B2 + r[3]*b1
        rdot[2] = (self.E[0]*self.B[1] - self.E[1]*self.B[0])/B2 + r[3]*b2
        rdot[3] = (self.E[0]*r[0] + self.E[1]*r[1] + self.E[2]*r[2])\
            /np.sqrt(B2)/rho
        rdot[4] = 0.
        rdot[5] = 0.
        rdot[6] = 0.

        return rdot
    #end def eom_GC

    def apply_BCs_periodic(self, grid):
        '''
        Wrap particle x-coordinate around for periodic BCs.

        Args:
            grid (Grid): grid object with which the particle is associated.

        Tests:
            >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
            >>> grid = Grid(5, 1.0, 1.0)
            >>> particle._initialize_6D(grid)
            >>> particle.r[0] = grid.length*1.5
            >>> particle.apply_BCs_periodic(grid)
            >>> particle.is_active()
            True
            >>> particle.r[0] == grid.length*0.5
            True
        '''
        self.r[0] = self.r[0]%(grid.length)
    #end def apply_BCs

    def apply_BCs_dirichlet(self, grid):
        '''
        Set particle to inactive when it's x-coordinate exceeds either wall in a
        dirichlet-dirichlet boundary condition case.

        Args:
            grid (Grid): grid object with which the particle is associated

        Tests:
            >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
            >>> grid = Grid(5, 1.0, 1.0)
            >>> particle._initialize_6D(grid)
            >>> particle.r[0] = grid.length + 1.0
            >>> particle.apply_BCs_dirichlet(grid)
            >>> particle.is_active()
            False
        '''
        if (self.r[0] < 0.0) or (self.r[0] > grid.length):
            self.active = 0
            self.at_wall = 1
        #end if
    #end def apply_BCs_dirichlet

    def reactivate(self, distribution, grid, time, p2c, m, charge_state, Z):
        '''
        Re-activate an inactive particle. This function pulls an r vector
        composed of x, y, z, v_x, v_y, v_z, t from a given distribution and
        applies it ot the particle before reactivating it. Additionally, the
        mass, charge, and p2c ratio can be reset at the instant of
        reactivation.

        Args:
            distribution (iterable): an iterable that returns a
                6-vector that overwrites the particle's coordinates
            grid (Grid): the grid object with which the particle is
                associated
            time (float): the particle's current time
            p2c (float): the ratio of computational to real particles
            m (float): particle mass
            charge_state (int): particle charge state
            Z (float): particle atomic number
        '''
        self.r = next(distribution)
        self.p2c = p2c
        self.m = m
        self.charge_state = charge_state
        self.Z = Z
        self.r[6] = time
        self.active = 1
        self.at_wall = 0
        self.from_wall = 0
        grid.add_particles(p2c)
    #end def reactivate
#end class Particle

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
        r[0] = np.random.normal(grid.length/2, grid.length/12.0)
        r[0] %= grid.length
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
        dist /= np.sum(dist)
        r[3] = -np.abs(np.random.choice(vels, p=dist)) + np.random.uniform(-1, 1)*(vels[1] - vels[0])/2.
        r[3] += vx
        if np.random.uniform(0, 1) < gamma:
            r[3] = vx_pert*vth
        yield r
    #end while
#end def flux_distribution_6D

class Grid:
    def __init__(self, ng, length, Te, bc='dirichlet-dirichlet'):
        self.ng = ng
        assert self.ng > 1, 'Number of grid points must be greater than 1'
        self.length = length
        assert self.length > 0.0, 'Length must be greater than 0'
        self.domain = np.linspace(0.0, length, ng)
        self.dx = self.domain[1] - self.domain[0]
        self.rho = np.zeros(ng)
        self.phi = np.zeros(ng)
        self.E = np.zeros(ng)
        self.n = np.zeros(ng)
        self.n0 = None
        self.rho0 = None
        self.Te = Te
        self.ve = np.sqrt(8./np.pi*kb*self.Te/me)
        self.added_particles = 0
        self.bc = bc
        if bc == 'dirichlet-dirichlet':
            self._fill_laplacian_dirichlet()
        elif bc == 'dirichlet-neumann':
            self._fill_laplacian_dirichlet_neumann()
            print(self.A)
        elif type(bc) != type(''):
            raise TypeError('bc must be a string')
        else:
            raise ValueError('Unimplemented boundary condition. Choose dirichlet-dirichlet or dirichlet-neumann')
    #end def __init__

    def __repr__(self):
        return f'Grid({self.ng}, {self.length}, {self.Te})'
    #end def __repr__

    def __len__(self):
        return int(self.ng)
    #end def __len__

    def copy(self):
        '''
        Copies a Grid object.

        Returns:
            grid (Grid): An equally-initialized Grid object.

        Notes:
            copy() will not copy weighting or other post-initialization
            calculations.

        Tests:
            >>> grid1 = Grid(2, 1.0, 1.0)
            >>> grid2 = grid1.copy()
            >>> grid1 == grid2
            False
            >>> grid1.ng == grid2.ng
            True
            >>> (grid1.A == grid2.A).all()
            True
        '''
        return Grid(self.ng, self.length, self.Te)
    #end def copy

    def weight_particles_to_grid_boltzmann(self, particles, dt):
        '''
        Weight particle densities and charge densities to the grid using a first
        order weighting scheme.

        Args:
            particles (list of Particles): list of particle objects
            dt (float): timestep; used for Boltzmann electron reference density
                update

        Tests:
            This test makes sure that particles are weighted correctly.
            >>> particle = Particle(1.0, 1.0, 1.0, 1.0, 1)
            >>> grid = Grid(101, 1.0, 1.0)
            >>> particle._initialize_6D(grid)
            >>> particle.x = 0.0
            >>> particle.r[0]
            0.0
            >>> particles = [particle]
            >>> grid.weight_particles_to_grid_boltzmann(particles, 1.0)
            >>> grid.n[0]
            100.0
            >>> particles[0].x = 1.0 - grid.dx/2
            >>> grid.weight_particles_to_grid_boltzmann(particles, 1.0)
            >>> round(grid.n[-1],6)
            50.0
        '''
        self.rho[:] = 0.0
        self.n[:] = 0.0

        for particle_index, particle in enumerate(particles):
            if particle.is_active():
                index_l = int(np.floor(particle.x/self.dx))
                index_r = (index_l + 1)
                w_r = (particle.x%self.dx)/self.dx
                w_l = 1.0 - w_r

                self.rho[index_l] += particle.charge_state*e*particle.p2c/self.dx*w_l
                self.rho[index_r] += particle.charge_state*e*particle.p2c/self.dx*w_r
                self.n[index_l] += particle.p2c/self.dx*w_l
                self.n[index_r] += particle.p2c/self.dx*w_r
            #end if
        #end for

        #if self.bc=='dirichlet-neumann':
        #    self.n[-1] *= 2
        #    self.rho[-1] *= 2

        if self.n0 == None: #This is only true for the first timestep.
            eta = np.exp(self.phi/self.Te/11600.)
            self.p_old = np.trapz(eta, self.domain)
            self.n0 = 0.9*np.average(self.n)
            self.rho0 = e*self.n0
        else:
            eta = np.exp(self.phi/self.Te/11600.)
            p_new = np.trapz(eta, self.domain)
            q_new = eta[0] + eta[-1]
            r_new = 2.*self.added_particles/dt
            fn = np.sqrt(self.ve*q_new*dt/p_new)
            self.n0 = self.n0*( (1.0 - fn)*self.p_old/p_new + fn - fn*fn/4.) + \
                r_new*dt/p_new
            self.rho0 = self.n0*e
            self.p_old = p_new
        #end if
    #end def weight_particles_to_grid_boltzmann

    def differentiate_phi_to_E_dirichlet(self):
        '''
        Find electric field on the grid from the negative differntial of the
        electric potential.

        Notes:
            Uses centered difference for interior nodes:

                d phi   phi[i+1] - phi[i-1]
            E = _____ ~ ___________________
                 dx            2 dx

            And forward difference for boundaries.

        Tests:
            >>> grid = Grid(6, 5.0, 1.0)
            >>> grid.phi[:] = 1.0
            >>> grid.differentiate_phi_to_E_dirichlet()
            >>> abs(grid.E)
            array([0., 0., 0., 0., 0., 0.])
            >>> grid.phi[:] = np.linspace(0.0, 1.0, 6)
            >>> grid.differentiate_phi_to_E_dirichlet()
            >>> grid.E
            array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])
        '''
        for i in range(1,self.ng-1):
            self.E[i] = -(self.phi[i + 1] - self.phi[i - 1])/self.dx/2.
        #end for
        self.E[0]  = -(self.phi[1]  - self.phi[0])/self.dx
        self.E[-1] = -(self.phi[-1] - self.phi[-2])/self.dx
    #end def differentiate_phi_to_E

    def _fill_laplacian_dirichlet(self):
        '''
        Internal method that creates the Laplacian matrix used to solve for the
        electric potential in dirichlet-dirichlet BCs.
        '''
        ng = self.ng

        self.A = np.zeros((ng,ng))

        for i in range(1,ng-1):
            self.A[i,i-1] = 1.0
            self.A[i,i]   = -2.0
            self.A[i,i+1] = 1.0
        #end for

        self.A[0, 0]  = 1.
        self.A[-1,-1] = 1.
    #end def fill_laplacian_dirichlet

    def _fill_laplacian_dirichlet_neumann(self):
        '''
        Internal method that creates the Laplacian matrix used to solve for the
        electric potential in dirichlet-neumann BCs.
        '''
        ng = self.ng

        self.A = np.zeros((ng, ng))

        for i in range(1,ng-1):
            self.A[i,i-1] = 1.0
            self.A[i,i]   = -2.0
            self.A[i,i+1] = 1.0
        #end for

        self.A[0,0] = 1.

        self.A[-1,-1] = 3.
        self.A[-1,-2] = -4.
        self.A[-1,-3] = 1.
    #end def

    def solve_for_phi(self):
        if self.bc == 'dirichlet-dirichlet':
            self.solve_for_phi_dirichlet_boltzmann()
        elif self.bc == 'dirichlet-neumann':
            self.solve_for_phi_dirichlet_neumann_boltzmann()
    #end def solve_for_phi

    def solve_for_phi_dirichlet(self):
        '''
        Solves Del2 phi = rho.

        Tests:
            >>> grid = Grid(5, 4.0, 1.0)
            >>> grid.rho[:] = np.ones(5)
            >>> grid.solve_for_phi_dirichlet()
            >>> list(grid.phi)
            [0.0, 1.5, 2.0, 1.5, 0.0]
        '''
        dx2 = self.dx*self.dx
        phi = np.zeros(self.ng)
        A = spp.csc_matrix(self.A)
        phi[:] = -sppla.inv(A).dot(self.rho)*dx2
        self.phi = phi - np.min(phi)
    #end def solve_for_phi_dirichlet

    def solve_for_phi_dirichlet_boltzmann(self):
        '''
        Solves for the electric potential from the charge density using
        Boltzmann (AKA adiabatic) electrons assuming dirichlet-dirichlet BCs.

        Tests:
            Tests are hard to write for the boltzmann solver. This one just
            enforces zero electric potential in a perfectly neutral plasma.
            >>> grid = Grid(5, 4.0, 1.0)
            >>> grid.n0 = 1.0/e
            >>> grid.rho[:] = np.ones(5)
            >>> grid.n[:] = np.ones(5)/e
            >>> grid.solve_for_phi_dirichlet_boltzmann()
            >>> grid.phi
            array([0., 0., 0., 0., 0.])
        '''
        residual = 1.0
        tolerance = 1e-9
        iter_max = 1000
        iter = 0

        phi = np.zeros(self.ng)
        D = np.zeros((self.ng, self.ng))

        dx2 = self.dx*self.dx
        c0 = e*self.n0/epsilon0
        c1 = e/kb/self.Te
        c2 = self.rho/epsilon0

        while (residual > tolerance) and (iter < iter_max):
            F = np.dot(self.A,phi) - dx2*c0*np.exp(c1*(phi)) + dx2*c2
            F[0] = phi[0]*0.
            F[-1] = phi[-1]*0.

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*phi))
            D[0,0] = -dx2*c0*c1
            D[-1,-1] = -dx2*c0*c1

            J = self.A + D
            J = spp.csc_matrix(J)
            #dphi = sppla.inv(J).dot(F)
            dphi, _ = sppla.bicgstab(J, F, x0=phi)

            phi = phi - dphi
            residual = dphi.dot(dphi)
            iter += 1
        #end while
        self.phi = phi - np.min(phi)
    #end def solve_for_phi_dirichlet

    def smooth_rho(self):
        rho_smooth = (np.roll(self.rho, -1) + 2.0 * self.rho + np.roll(self.rho, 1)) * 0.25
        rho_smooth[0] = self.rho[0]
        rho_smooth[-1] = self.rho[-1]
        self.rho = rho_smooth
    #end def smooth_field_p

    def solve_for_phi_dirichlet_neumann_boltzmann(self):
        '''
        Solves for the electric potential from the charge density using
        Boltzmann (AKA adiabatic) electrons assuming dirichlet-neumann BCs.

        Tests:
            Tests are hard to write for the boltzmann solver. This one just
            enforces zero electric potential in a perfectly neutral plasma.
            >>> grid = Grid(5, 4.0, 1.0)
            >>> grid.n0 = 1.0/e*epsilon0
            >>> grid.rho[:] = np.ones(5)
            >>> grid.n[:] = np.ones(5)/e*epsilon0
            >>> grid.solve_for_phi_dirichlet_neumann_boltzmann()
            >>> grid.phi
            array([0., 0., 0., 0., 0.])
        '''
        residual = 1.0
        tolerance = 1e-3
        iter_max = 100
        iter = 0

        phi = self.phi
        D = np.zeros((self.ng, self.ng))

        dx2 = self.dx*self.dx
        c0 = e*self.n0/epsilon0
        c1 = e/kb/self.Te
        c2 = e*self.n/epsilon0

        while (residual > tolerance) and (iter < iter_max):
            F = np.dot(self.A,phi) - dx2*c0*np.exp(c1*(phi)) + dx2*c2
            F[0] = phi[0]
            F[-1] = 0.

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi)))
            D[0,0] = -dx2*c0*c1
            D[-1,-1] = 0.

            J = self.A + D
            J = spp.csc_matrix(J)
            dphi = sppla.inv(J).dot(F)

            phi = phi - dphi
            residual = la.norm(dphi)
            iter += 1
        #end while
        self.phi = phi - np.min(phi)
    #end def solve_for_phi_dirichlet_neumann

    def reset_added_particles(self):
        self.added_particles = 0
    #end def reset_added_particles

    def add_particles(self, particles):
        self.added_particles += 2.*particles
    #end def add_particles
#end class Grid

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
                particle.interpolate_electric_field_dirichlet(grid)
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
                particle.interpolate_electric_field_dirichlet(grid)
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
                particle.interpolate_electric_field_dirichlet(grid)
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
