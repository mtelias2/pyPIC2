import scipy as sp
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from constants import *

class Fluid:
    def __init__(self, L, psi, vpp, Ti, Te, n0, u0, omega_c, omega, mi, num_cycles, num_steps_per_gyro=500):
        self.ng = 2*L
        self.L = L
        self.psi = psi
        self.Ti = Ti
        self.Te = Te
        self.u0 = u0
        self.n0 = n0
        self.vpp = vpp

        self.t = 0

        self.bx = np.cos(psi)
        self.by = np.sin(psi)

        self.omega_c = omega_c
        self.omega = omega

        self.num_timesteps_per_cycle = np.int(np.ceil(num_steps_per_gyro/omega))
        self.num_cycles = num_cycles
        self.debye = np.sqrt((Te*e)*epsilon0/e/e/n0)
        #self.L = self.num_debye*self.debye
        self.dx = self.L/self.ng
        self.cs = np.sqrt(Te*e/mi)
        self.omega_p = np.sqrt(n0*e*e/mi/epsilon0)
        self.meu = np.sqrt(mi/(me*e*np.pi))
        self.B = self.omega_p*omega_c*mi/e

        self.num_timesteps = self.num_timesteps_per_cycle*self.num_cycles

        # self.ni = np.ones((self.num_timesteps, ng))
        # self.ux = self.bx*u0*np.ones((self.num_timesteps, ng))
        # self.uy = self.by*u0*np.ones((self.num_timesteps, ng))
        # self.uz = np.zeros((self.num_timesteps, ng))
        # self.phi = np.ones((self.num_timesteps, ng))

        self.ni = np.ones((self.num_timesteps, self.ng))
        self.ux = self.bx*u0*np.ones((self.num_timesteps, self.ng))
        self.uy = self.by*u0*np.ones((self.num_timesteps, self.ng))
        self.uz = np.zeros((self.num_timesteps, self.ng))
        self.phi = np.ones((self.num_timesteps, self.ng))
        self.E = np.zeros((self.num_timesteps, self.ng))


        self.dt = 2.*np.pi/self.omega/self.num_timesteps_per_cycle

        self.A = np.diag(-2*np.ones(self.ng)) + np.diag(np.ones(self.ng - 1), 1) + np.diag(np.ones(self.ng - 1), -1)
        self.A[0, 0] = -3.
        self.A[0, 1] = 4.
        self.A[0, 2] = -1.
        self.A[-1, -1] = 1.
        self.A[-1, -2] = 0.
    #end def __init__

    def solve_ni(self):

        t = self.t

        ni1 =  self.ni[t, :]

        for n in range(1, self.ng):
            a = self.dt/self.dx*(self.ni[t, n]*self.ux[t, n] - self.ni[t, n - 1]*self.ux[t, n - 1])
            ni1[n] = self.ni[t, n] - a
        #end for

        self.ni[t + 1, :] = ni1

    #end def solve_ni

    def solve_momentum(self):

        t = self.t

        ux1 = np.zeros(self.ng)
        uy1 = np.zeros(self.ng)
        uz1 = np.zeros(self.ng)

        ux1[0] = self.ux[t, 0]
        uy1[0] = self.uy[t, 0]
        uz1[0] = self.uz[t, 0]

        for n in range(1, self.ng):
            #UxK(n)=Ux(n) -Delt*Ux(n)*(Ux(n)-Ux(n-1))/Delx + (-Delt*Efield(n)) -( Delt*omegaC*Uz(n)*by)
            ux1[n] = self.ux[t, n] - self.dt*self.ux[t, n]*(self.ux[t, n] - self.ux[t, n - 1])/self.dx + \
                (-self.dt*self.E[t, n]) - self.dt*self.omega_c*self.uz[t, n]*self.by

            #UyK(n)=Uy(n) - Delt*Ux(n)*(Uy(n)-Uy(n-1))/Delx +  Delt*omegaC*Uz(n)*bx
            uy1[n] = self.uy[t, n] - self.dt*self.ux[t, n]*(self.uy[t, n] - self.uy[t, n - 1])/self.dx + \
                self.dt*self.omega_c*self.uz[t, n]*self.bx

            #UzK(n)=Uz(n) - Delt*Ux(n)*(Uz(n)-Uz(n-1))/Delx + Delt*omegaC*(Ux(n)*by - Uy(n)*bx)
            uz1[n] = self.uz[t, n] - self.dt*self.ux[t, n]*(self.uz[t, n] - self.uz[t, n - 1])/self.dx + \
                self.dt*self.omega_c*(self.ux[t, n]*self.by - self.uy[t, n]*self.bx)

        #end for

        self.ux[t + 1, :] = ux1
        self.uy[t + 1, :] = uy1
        self.uz[t + 1, :] = uz1

    #end def solve_momentum

    def solve_phi(self):

        residual = 1
        tolerance = 1e-6
        iter = 0
        iter_max = 100

        B = self.vpp*np.cos(self.omega*self.t*self.dt)/2. #???
        t = self.t

        #BC[0] = 0.
        #BC[self.ng] = -B #???????

        # There's a whole section here that I don't understand
        # Ah, this is for the RF potential I guess?
        if self.t / self.num_timesteps_per_cycle < 0.5:
            phi0 = -np.log(self.u0 / (self.meu*self.bx*np.cosh(B)))
        else:
            T2 = int(np.floor(self.t - self.num_timesteps_per_cycle/2))
            phi0 = -np.log( (self.ni[t - 1, N]*self.ux[t - 1, N]+ self.ni[t - T2, N]*self.ux[t - T2, N] )/(self.meu*self.bx*(np.exp(B) + np.exp(-B))))
        #end if

        phi1 = phi0*np.ones(self.ng)

        dx2 = self.dx*self.dx

        BC = np.zeros(self.ng)
        BC[0] = 0.
        BC[-1] = -B

        while (residual > tolerance) and (iter < iter_max):

            J = self.A - np.diag(np.exp(phi1 - phi0))*dx2

            F = self.A.dot(phi1) - np.exp(phi1-phi0)*dx2 + self.ni[t, :]*dx2 - BC

            J = spp.csc_matrix(J)
            dphi, _ = sppla.bicgstab(J, F, x0=phi1)
            phi1 = phi1 - dphi

            residual = dphi.dot(dphi)
            iter += 1
        #end while

        #phi1 = phi1 - np.min(phi1)

        self.phi[t + 1, :] = phi1

    #end def solve_phi

    def differentiate_phi_to_E(self):

        t = self.t

        E1 = self.E[t - 1, :]

        for n in range(1, self.ng - 1):
            E1[n] = -(self.phi[t + 1, n + 1] - self.phi[t + 1, n - 1])/self.dx/2.
        #end for

        E1[0] = -(self.phi[t + 1, 1]- self.phi[t + 1, 0])/self.dx
        E1[-1] = -(self.phi[t + 1, -1] - self.phi[t + 1, -2])/self.dx

        self.E[t + 1, :] = E1
    #end def differentiate_phi_to_E

    def advance(self):
        self.solve_phi()
        self.differentiate_phi_to_E()
        self.solve_momentum()
        self.solve_ni()
        self.t += 1

def main():
    L = 100
    num_debye = 100
    psi = np.arccos(1)
    vpp = 200 #peak-peak voltage
    Ti = 3
    Te = 3
    n0 = 1e18
    u0 = 1.1
    omega_c = 0.1
    omega = 1.
    mi = mp
    num_cycles = 8
    f = Fluid(L, psi, vpp, Ti, Te, n0, u0, omega_c, omega, mi, num_cycles)

    plt.ion()
    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)

    delay = 0.001

    for i in range(100):
        f.advance()

        plt.figure(1)
        plt.clf()
        plt.title('Phi')
        plt.plot(f.phi[f.t, :])
        plt.pause(delay)

        plt.figure(2)
        plt.clf()
        plt.title('E')
        plt.plot(f.E[f.t, :])
        plt.pause(delay)

        plt.figure(3)
        plt.clf()
        plt.title('U')
        plt.plot(f.ux[f.t, :])
        plt.plot(f.uy[f.t, :])
        plt.plot(f.uz[f.t, :])
        plt.pause(delay)

        plt.figure(4)
        plt.clf()
        plt.title('Ni')
        plt.plot(f.ni[f.t, :])
        plt.pause(delay)

if __name__ == '__main__':
    main()
