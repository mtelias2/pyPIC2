import numpy as np
from constants import *

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


            if (grid.Ion_flux_right != None) or (grid.Ion_flux_left != None):

                ind = int(np.floor(self.x/grid.dx))
                w_l = (self.x%grid.dx)/grid.dx
                w_r = 1.0 - w_l

                if ind == 0: #left boundary
                    grid.Ion_flux_left+=(self.r[3]*w_l*self.p2c/(0.5*grid.dx))

                if ind == (grid.ng-2): #right boundary
                    grid.Ion_flux_right+=(self.r[3]*w_r*self.p2c/(0.5*grid.dx))

            #this section is needed for calculations of reference density Elias method
            #def Elias in Grid.py



        #end if
    #end def apply_BCs_dirichlet

    def apply_BCs_dirichlet_reflection(self, grid):
        if self.r[0] < 0:
            self.active = 0
            self.at_wall = 1
        elif self.r[0] > grid.length:
            self.r[3] = -self.r[3]
            self.r[0] = grid.length - (self.r[0] - grid.length)

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
