import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import numpy as np
import scipy.linalg as la
from constants import *
import utils

class Grid:
    def __init__(self, ng, length, Te, density, dt,omega,RF_amptitude,alpha, bc='dirichlet-dirichlet'):
        '''
        This function creates the grid space for the simulations
        Args:
            ng          total number of grid grid points
            length      Total length of simulation in physical units
            Te          Election temperature
            bc          Boundary condition type
                    currently on Drichlet Driechlet
        '''
        self.ng = ng
        assert self.ng > 1, 'Number of grid points must be greater than 1'
        self.length = length
        assert self.length > 0.0, 'Length must be greater than 0'
        self.domain = np.linspace(0.0, length, ng)

        self.dx = self.domain[1] - self.domain[0]
        #assuming uniform grid space
        self.dt=dt
        self.density=density

        self.rho = np.zeros(ng)#i think density difference of charges

        self.phi = np.zeros(ng) #electric potential

        self.E = np.zeros(ng) #electric field

        self.n = np.zeros(ng)
        self.n0 = None
        self.rho0 = None
        self.Te = Te
        self.ve = np.sqrt(8./np.pi*kb*self.Te/me)
        self.added_particles = 0
        self.bc = bc

        self.alpha=alpha

        self.omega=omega
        self.RF_amptitude=RF_amptitude

        # things needed for Elias algorithm might change them later
        self.Ion_flux_right=None #at index L
        self.Ion_flux_left=None  #at index 0

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

    def weight_particles_to_grid_boltzmann(self, particles):
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


    def reference_density_update(self,time,method="Hagelaar"):
        '''
        This function updates the reference density.
        3 methods will be programed.

        3 other functions will be called depending on the method.

        Hagelaar 2008
        kwok 2008

        Args:
            Ion density profile
            Previous time step total number of electrons
            Reference density update method

        No tests programed yet

        '''
        #Set the reference density for the boltzmann equation
        self.phi0=0
        if method=='Hagelaar':
            self._Hag()
        #elif method=='Kwok':
        #    self._Kwok()
        elif method=='Elias':
            self._Elias(time)

    def _Hag(self):
        if self.n0 == None: #This is only true for the first timestep.
            eta = np.exp(self.phi/self.Te/11600.)
            self.p_old = np.trapz(eta, self.domain)
            self.n0 = 0.9*self.density
            self.rho0 = e*self.n0
        else:
            eta = np.exp(self.phi/self.Te/11600.)
            p_new = np.trapz(eta, self.domain)
            q_new = eta[0] + eta[-1]
            r_new = 2.*self.added_particles/self.dt
            fn = np.sqrt(self.ve*q_new*self.dt/p_new)
            self.n0 = self.n0*( (1.0 - fn)*self.p_old/p_new + fn - fn*fn/4.) + \
                r_new*self.dt/p_new
            self.rho0 = self.n0*e
            self.p_old = p_new

    #def _Kwok(self):
    #    if self.n0 == None: #This is only true for the first timestep.
    #        eta = np.exp(self.phi/self.Te/11600.)
    #        self.p_old = np.trapz(eta, self.domain)
    #        self.n0 = 0.9*self.density
    #        self.rho0 = e*self.n0
    #        Ne_old=self.density*self.length
    #    else:

            #new density is old one - total numebr of particles lost
    #        Ne_new = Ne_old




            #copying the old density
            #Ne(t)=Ne(t-1) - v Int_walls(ne(x)dx)
    #        Ne_old=Ne_new - (self.ve/4 * ( ) )


    #to be defined my method. This method cannot be used for EM codes though.
    def _Elias(self,time):
        """
        This method is based on Elias 2020.
        The codes used are to fit the equaiton specified by the paper
        Each term is explained but the equations are not.
        """

        if self.n0 == None: #This is only true for the first timestep.
            self.n0 = 0.9*self.density
            self.rho0 = e*self.n0
            ##Things needed for charge conservation in Elias methods
            self.Ion_flux_right=0 #at index L
            self.Ion_flux_left=0  #at index 0
        else:

            Ji=0  # This term represents the total ion currents at the walls at time step t
            Ue=0  # This term represents the electron fluid velocity needed to calculate n_0
            Jd=0  # This term represents the displacement current, I am having troubles including it. and slight doubts about its validity
            #Jd will be included and tested later otherwise i do not get my phd

            #left and right Boundary conditions just to ease calculations not needed explicitly
            LBC=self.RF_amptitude*np.sin(self.omega*time)
            RBC=self.RF_amptitude*np.sin(self.omega*time+np.pi)

            Ji=self.Ion_flux_right-self.Ion_flux_right #difference between right and left

            #equation for Ue might need to be changed check what gives you the right value there is a 3/2 missing somewhere
            Ue=self.ve*np.cos(self.alpha)*(np.exp(RBC/self.Te/11600.)+np.exp(LBC/self.Te/11600.))/(np.sqrt(np.pi*2))

            self.n0=Ji/Ue

            #reseting for the next iteration
            self.Ion_flux_left=0.
            self.Ion_flux_right=0.



    def differentiate_phi_to_E_dirichlet(self):
        '''
        Find electric field on the grid from the negative differntial of the
        electric potential.

        Notes:
            Uses centered difference for interior nodes:

                d phi   phi[i+1] - phi[i-1]
            E = _____ ~ ___________________
                 dx            2 dx

            And forward/backward difference for boundaries.

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

        #Setting the BC
        self.BC0=0
        self.BC1=0

        phi = np.zeros(self.ng)
        D = np.zeros((self.ng, self.ng))

        dx2 = self.dx*self.dx
        c0 = e*self.n0/epsilon0
        c1 = e/kb/self.Te
        c2 = self.rho/epsilon0

        while (residual > tolerance) and (iter < iter_max):

            F = np.dot(self.A,phi)
            for i in range(1,self.ng-1):
                F[i]+= -dx2*c0*np.exp(c1*(phi[i]-self.phi0)) + dx2*c2[i]

            F[0]  -= self.BC0
            F[-1] -= self.BC1

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi-self.phi0)))

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

        #Setting the BC
        self.BC0=0
        self.BC1=0

        dx2 = self.dx*self.dx
        c0 = e*self.n0/epsilon0
        c1 = e/kb/self.Te
        c2 = e*self.n/epsilon0

        while (residual > tolerance) and (iter < iter_max):
            F = np.dot(self.A,phi)
            for i in range(1,self.ng-1):
                F[i]+= -dx2*c0*np.exp(c1*(phi[i]-self.phi0)) + dx2*c2[i]

            F[0]  -= self.BC0
            F[-1] -= self.BC1

            np.fill_diagonal(D, -dx2*c0*c1*np.exp(c1*(phi-self.phi0)))

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
