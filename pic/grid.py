import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import numpy as np
import scipy.linalg as la

from constants.constants import *

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
        if self.bc == 'dirichlet-dirichlet':
            self._fill_laplacian_dirichlet()
        elif self.bc == 'dirichlet-neumann':
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

            >>> from pic.particle import Particle
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

        if self.bc=='dirichlet-neumann':
            self.n[-1] *= 2
            self.rho[-1] *= 2

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

    def differentiate_phi_to_E(self):
        #in theory, different BCs should have different differentiation, but
        #currently it doesn't matter
        self.differentiate_phi_to_E_dirichlet()
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

    def solve_for_phi(self):
        if self.bc == 'dirichlet-neumann':
            self.solve_for_phi_dirichlet_neumann_boltzmann()
        elif self.bc =='dirichlet-dirichlet':
            self.solve_for_phi_dirichlet_boltzmann()

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
