import numpy as np
import time
import copy
from scipy import interpolate, special
import scipy.sparse.linalg as linalg
from numpy.polynomial.legendre import leggauss
###################################################################################################
# obtain fission spectrum for U-235, E must be in (MeV) 
default_chi = lambda E: 0.4865*np.sinh(np.sqrt(2*E))*np.exp(-E)
###################################################################################################
# obtain speed in (cm/s) for a particular energy (MeV) 
vel = lambda E: np.sqrt(2.*E/938.280)*3e10 
###################################################################################################
class ZoneSpatialMesh:
    def __init__(self, mat, x_lower_bound, x_upper_bound, num_cells=1, log_option=False):
        self = self
        self.mat = mat
        self.x_lower_bound = np.float(x_lower_bound)
        self.x_upper_bound = np.float(x_upper_bound)
        self.num_cells = num_cells
        self.log_option = log_option
        if self.log_option == True:
            self.x_edges = np.logspace(np.log10(self.x_lower_bound), np.log10(self.x_upper_bound), num=self.num_cells+1)
        else:
            self.x_edges = np.linspace(self.x_lower_bound, self.x_upper_bound, num=self.num_cells+1)
        self.x_mids = 0.5*(self.x_edges[1:] + self.x_edges[:-1])
        self.dx = self.x_edges[1:] - self.x_edges[:-1]

    def update_discretization(self, N, option):
        self.num_cells = N
        if self.log_option == True:
            self.x_edges = np.logspace(np.log10(self.x_lower_bound), np.log10(self.x_upper_bound), num=self.num_cells+1)
        else:
            self.x_edges = np.linspace(self.x_lower_bound, self.x_upper_bound, num=self.num_cells+1)
        self.x_mids = 0.5*(self.x_edges[1:] + self.x_edges[:-1])
        self.dx = self.x_edges[1:] - self.x_edges[:-1]

    def print_zone(self):
        print self.mat
        print "local number of spatial cells:", self.num_cells
        print self.x_edges
###################################################################################################
class GlobalMesh:
    def __init__(self, mat_dict, zones, num_angles, num_c_grps, geom):
        self = self
        self.geom = geom
        self.num_zones = len(zones)
        self.zones = zones  

        self.cell_zone_number = []
        for zone_number in range(len(self.zones)):
            for i in range(self.zones[zone_number].num_cells):
                self.cell_zone_number.append( zone_number )

        self.global_to_local = []
        for zone_number in range(len(self.zones)):
            for i in range(self.zones[zone_number].num_cells):
                self.global_to_local.append( i )
        self.num_cells = len(self.global_to_local)  

        self.cell_mat = []
        for zone_number in range(len(self.zones)):
            for i in range(self.zones[zone_number].num_cells):
                self.cell_mat.append( self.zones[zone_number].mat ) 

        self.x_edges = list(self.zones[0].x_edges)
        for zone in self.zones[1:]:
            self.x_edges += list(zone.x_edges[1:])
        self.x_edges = np.array(self.x_edges)
        self.x_mids = 0.5*(self.x_edges[1:] + self.x_edges[:-1])
        self.dx = self.x_edges[1:] - self.x_edges[:-1] 
        self.V = np.copy(self.dx)

        self.A_left = np.ones( self.num_cells )
        self.A_right = np.ones( self.num_cells )

        self.mat_dict = mat_dict
        self.num_grps = mat_dict[mat_dict.keys()[0]].num_grps
        self.emid = mat_dict[mat_dict.keys()[0]].emid
        self.de = mat_dict[mat_dict.keys()[0]].de
        self.nlgndr = mat_dict[mat_dict.keys()[0]].nlgndr
        
        for mat in sorted(mat_dict.keys()):
            if self.num_grps != mat_dict[mat].num_grps:
                print "ERROR: 'num_grps' is inconsitent for different materials"
            if np.linalg.norm(self.emid - mat_dict[mat].emid) > 1e-13:
                print "ERROR: 'emid' is inconsitent for different materials"
            if np.linalg.norm(self.de - mat_dict[mat].de) > 1e-13:
                print "ERROR: 'de' is inconsitent for different materials"

        self.num_angles = num_angles
        self.mu, self.w = leggauss(num_angles)
        self.legendre = np.zeros((self.nlgndr,self.num_angles))
        self.legendre_and_normalization = np.zeros((self.nlgndr,self.num_angles))
        # precomputing legendre polynomial value for order ell for direction mu_m
        for moment in range(self.nlgndr):
            for m in range(self.num_angles):
                self.legendre[moment,m] = special.legendre(moment)(self.mu[m]) 
                self.legendre_and_normalization[moment,m] = (2.*moment+1.)/2.*self.legendre[moment,m]
        
        # ----------------------------- for spherical code: ---------------------------------------
        if self.geom == "sphere":
            self.x_left = self.x_mids - 0.5*self.dx
            self.x_right = self.x_mids + 0.5*self.dx

            I = len(self.x_mids)
            self.V = np.zeros(I) 
            self.A_left = np.zeros(I) 
            self.A_right = np.zeros(I) 
            self.gamma = np.zeros(I) 
            self.gamma_hat = np.zeros(I) 

            for i in range(I):
                self.V[i] = 4./3.*np.pi*(self.x_right[i]**3 - self.x_left[i]**3)
                self.A_left[i] = 4.*np.pi*self.x_left[i]**2
                self.A_right[i] = 4.*np.pi*self.x_right[i]**2
                self.gamma[i] = (3./4.*self.x_right[i]**4 - self.x_left[i]*self.x_right[i]**3 + 1./4.*self.x_left[i]**4) / (self.dx[i]*(self.x_right[i]**3 - self.x_left[i]**3))
                self.gamma_hat[i] = (2./3.*self.x_right[i]**3 - self.x_left[i]*self.x_right[i]**2 + 1./3.*self.x_left[i]**3) / (self.dx[i]*(self.x_right[i]**2 - self.x_left[i]**2))

            self.mu_down = np.zeros(self.num_angles)
            self.mu_up = np.zeros(self.num_angles)
            self.alpha_down = np.zeros(self.num_angles)
            self.alpha_up = np.zeros(self.num_angles)
            self.beta = np.zeros(self.num_angles)
            for m in range(self.num_angles):
                if m == 0:
                    self.mu_down[0] = -1
                    self.alpha_down[0] = 1. - self.mu_down[0]**2.
                else:
                    self.mu_down[m] = np.copy(self.mu_up[m-1])
                    self.alpha_down[m] = np.copy(self.alpha_up[m-1])
                self.mu_up[m] = self.mu_down[m] + self.w[m]
                self.alpha_up[m] = self.alpha_down[m] - 2.*self.mu[m]*self.w[m]
                self.beta[m] = (self.mu[m] - self.mu_down[m])/self.w[m]
        # -----------------------------------------------------------------------------------------

        self.mat_sigt = {}
        self.mat_sigf = {}
        self.mat_p = {}
        self.mat_invSpgrp = {}
        self.mat_chid = {}
        self.mat_beta_fnubar = {}
        self.num_precursors = mat_dict[mat].num_precursors
        self.mat_sigf_MT18 = {}

        for mat in sorted(mat_dict.keys()):
            self.mat_sigt[mat] = mat_dict[mat].sigt
            self.mat_sigf[mat] = mat_dict[mat].sigf
            self.mat_p[mat] = mat_dict[mat].p  
            self.mat_invSpgrp[mat] = mat_dict[mat].invSpgrp
            if '18' in mat_dict[mat].crossSectionMTList: 
                self.mat_sigf_MT18[mat] = mat_dict[mat].sigf_MT18
                self.mat_chid[mat] = mat_dict[mat].pdt_chid
                self.mat_beta_fnubar[mat] = mat_dict[mat].pdt_beta_fnubar
            #else:
            #    self.mat_sigf_MT18[mat] = mat_dict[mat].sigf_tot

        self.sigt        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.sigt[:,i]          = self.mat_sigt[self.cell_mat[i]]

        self.invSpgrp        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.invSpgrp[:,i]          = self.mat_invSpgrp[self.cell_mat[i]]

        self.num_c_grps = num_c_grps
        self.num_grps = len(self.emid)
        sub_g = int(np.ceil( self.num_grps / self.num_c_grps ))

        self.mat_sigs = {}
        self.mat_D = {}
        self.mat_siga = {}
        for mat in sorted(mat_dict.keys()):
            self.mat_sigs[mat] = np.zeros((self.nlgndr,self.num_grps))
            for gp in range(self.num_grps):
                for g in range(self.num_grps):
                    for moment in range(self.nlgndr):
                        self.mat_sigs[mat][moment][gp] += self.mat_p[mat][moment][gp,g]

            self.mat_D[mat] = np.zeros(self.num_grps)
            for gp in range(self.num_grps):
                if len(self.mat_sigs[mat]) > 1:
                    self.mat_D[mat][gp] = 1./(3.*(self.mat_sigt[mat][gp] - self.mat_sigs[mat][1][gp]))
                else:
                    self.mat_D[mat][gp] = 1./(3.*self.mat_sigt[mat][gp])

            self.mat_siga[mat] = np.zeros(self.num_grps)
            for gp in range(self.num_grps):
                self.mat_siga[mat][gp] = self.mat_sigt[mat][gp] - self.mat_sigs[mat][0][gp]

        include_delayed_neutrons = True
        if include_delayed_neutrons == True:
            for mat in sorted(mat_dict.keys()):
                for gp in range(self.num_grps):
                    for g in range(self.num_grps):
                        for d in range(self.num_precursors):
                            self.mat_sigf[mat][gp,g] += self.mat_chid[mat][d,g]*self.mat_beta_fnubar[mat][gp]*self.mat_sigf_MT18[mat][gp]

        self.mat_coarse_sigf = {}
        self.mat_coarse_p = {}
        for mat in sorted(mat_dict.keys()):
            self.mat_coarse_sigf[mat] = np.zeros((self.num_c_grps,self.num_c_grps))
            self.mat_coarse_p[mat] = np.zeros((self.nlgndr,self.num_c_grps,self.num_c_grps))

            for gp in range(self.num_grps):
                cgp = int(np.floor(gp/sub_g))
                for g in range(self.num_grps):
                    cg = int(np.floor(g/sub_g))
                    self.mat_coarse_sigf[mat][cgp,cg] += self.mat_sigf[mat][gp,g]
                    for moment in range(self.nlgndr):
                        self.mat_coarse_p[mat][moment][cgp,cg] += self.mat_p[mat][moment][gp,g]  
                        #print moment, cgp, cg, gp, g, sub_g 
         
        self.mapping = []
        prev_cg = -1
        for g in range(self.num_grps):
            print g, prev_cg
            cg = int(np.floor(g/sub_g))            
            if cg != prev_cg:
                self.mapping.append([g])
            else:
                self.mapping[-1].append(g)
            prev_cg = np.copy(cg)

    def print_space(self, v=0):
        print "\nTotal number of spatial cells:", self.num_cells
        print "\n  x_edges = ", self.x_edges
        print "\n  x_mids = ", self.x_mids
        print "\n  dx = ", self.dx

        if v > 0:
            for zone_number in range(len(self.zones)):
                zone = self.zones[zone_number]
                print "\nZone #"+str(zone_number)
                print "--------------------------------------------------------------------------"
                zone.print_zone()
                print "--------------------------------------------------------------------------"
        
        if v > 1:
            print
            for i in range(self.num_cells):
                print 'global cell '+str(i)+' ('+self.cell_mat[i]+') --> x=['+str(self.x_edges[i])+','+str(self.x_edges[i+1])+']'

    def print_angles(self, v=0):
        print "\nNumber of angles:", self.num_angles
        print "\n  mu = ", self.mu
        print "\n  w = ", self.w

        if v > 0:
            print "\n  P_l(mu) = ", self.legendre

    def print_energies(self, v=0):
        print "\nNumber of groups:", self.num_grps
        print "\n  emid = ", self.emid
        print "\n  de = ", self.de
###################################################################################################
class Physics:
    def __init__(self, mesh, L_max=8):
        self = self
        L = min(L_max, mesh.nlgndr)
        self.phi = np.zeros((mesh.num_grps, L, mesh.num_cells)) 
        self.phi[:,0,:] += np.ones((mesh.num_grps, mesh.num_cells))

        self.c_phi = np.zeros((mesh.num_c_grps, L, mesh.num_cells)) 
        self.c_phi[:,0,:] += np.ones((mesh.num_c_grps, mesh.num_cells))

        self.psi = (1./mesh.num_angles)*np.ones((mesh.num_grps, mesh.num_angles, mesh.num_cells))
        self.psi_left = (1./mesh.num_angles)*np.ones((mesh.num_grps, mesh.num_angles, mesh.num_cells))
        self.psi_right = (1./mesh.num_angles)*np.ones((mesh.num_grps, mesh.num_angles, mesh.num_cells))

        self.k = 1.
        self.alpha = 0.
###################################################################################################
class BoundaryCondition:
    def __init__(self, mesh, left='vacuum', right='vacuum', phi_left=0, phi_right=0, J_left=0, J_right=0, psi_left_mu1=0, psi_right_mu1=0, time_profile=0):
        self = self
        self.left = left
        self.right = right  

        G, N = mesh.num_grps, mesh.num_angles
        self.psi_left = np.zeros((G, N))  
        self.psi_right = np.zeros((G, N))  

        if left == 'isotropic':
            self.psi_left[:,N/2:N] = phi_left / sum( mesh.w )
        if right == 'isotropic':
            self.psi_right[:,N/2] = phi_boundary / sum( mesh.w )

        if left == "partial_current":
            self.psi_left[:,N/2:N] = J_left / sum( mesh.w[N/2:N] * mesh.mu[N/2:N] ) * np.ones(N/2)
        if right == "partial_current":
            self.psi_right[:,0:N/2] = -J_right / sum( mesh.w[0:N/2] * mesh.mu[0:N/2] ) * np.ones(N/2)

        if left == 'mu1':
            self.psi_left[:,N-1] = (1. - mu[N-1])/(mu[N-1] - mu[N-2]) * psi_left_mu1 / w[N-1]
        if right == 'mu1':
            self.psi_right[:,0] = (1. + mu[0])/(-mu[0] + mu[1]) * psi_right_mu1 / w[0] 
###################################################################################################     
def power_iterations(mesh, bc, problem_type, discretization, mode='normal', L_max=9, S_tol=1e-6, F_tol=1e-6, max_its=1e3, tol=1e-6, talk=True, k_exact=None, recomp_F=1, recomp_S=[1,1,1,1,1,1,1,1,1], DSA_opt=True):
    G, N, I = mesh.num_grps, mesh.num_angles, mesh.num_cells
    L = min(mesh.nlgndr, L_max)
    #----------------------------------------------------------------------------------------------
    def dot(a,b):
        ab = 0
        for i in range(len(a)):
            ab += a[i]*b[i]
        return ab   
    #----------------------------------------------------------------------------------------------
    def take_moments(psi):
        phi = np.zeros((G, L, I))
        for g in range(G):
            for moment in range(L):
                for m in range(N):
                    for i in range(I):
                        phi[g,moment,i] += psi[g,m,i]*mesh.legendre[moment,m]*mesh.w[m]
        return phi   
    #----------------------------------------------------------------------------------------------
    def inner_prod(x,Ax):
        inner_prod = 0
        for g in range(G):
            for m in range(N):
                for i in range(I):
                    inner_prod += x[g,m,i]*Ax[g,m,i]*mesh.V[i]*mesh.w[m]*mesh.de[g]
        return inner_prod  
    #----------------------------------------------------------------------------------------------
    def invV(N=0):
        if N==0:
            return mesh.invSpgrp
        else:
            invV = np.zeros((G, N, I))
            for g in range(G):
                for m in range(N):
                    for i in range(I):
                        invV[g,m,i] = mesh.invSpgrp[g,i]
            return invV
    #----------------------------------------------------------------------------------------------
    def invH_slab(Q, t_abs=0.):
        # sweeps across all cells for all energies and directions
        psi = np.zeros((G,N,I)); psi_left = np.zeros((G,N,I)); psi_right = np.zeros((G,N,I))
        sigt = mesh.sigt + t_abs # adding time absorption
        for g in range(G):
            for m in range(N):
                psi_L = 0.
                psi_R = 0.
                psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:] = sweep_slab_diamond_difference(I, mesh.dx, mesh.mu[m], sigt[g,:], Q[g,m,:], psi_L, psi_R)
        return take_moments(psi), psi, psi_left, psi_right
    #----------------------------------------------------------------------------------------------
    def invH_sphere(Q, t_abs=0.):
        # sweeps across all cells for all energies and directions
        psi = np.zeros((G,N,I)); psi_left = np.zeros((G,N,I)); psi_right = np.zeros((G,N,I));
        psi_hat_down = np.zeros((G,N,I)); psi_hat_up = np.zeros((G,N,I))
        mu = mesh.mu
        psi_L = np.zeros(G); psi_R = np.zeros(G)
        sigt = mesh.sigt + t_abs # adding time absorption
        for g in range(G):
            psi_R[g] = 0.
            for m in range(N):
                if m == 0:
                    Q_start = Q[g,0,:]*(mu[1]+1.)/(mu[1]-mu[0]) - Q[g,1,:]*(1.+mu[0])/(mu[1]-mu[0])
                    psi_L[g], psi_hat_down[g,0,:] = sweep_starting_direction_diamond_difference(mesh, sigt, g, Q_start, psi_R[g])
                else:
                    psi_hat_down[g,m,:] = np.copy(psi_hat_up[g,m-1,:])
                psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:], psi_hat_up[g,m,:] = sweep_sphere_diamond_difference(mesh, g, m, Q[g,m,:], psi_hat_down[g,m,:], psi_L[g], psi_R[g])

        return take_moments(psi), psi, psi_left, psi_right
    #----------------------------------------------------------------------------------------------
    def invH_inf(Q, t_abs=0.):
        # infinite-medium Boltzmann operator (no streaming)
        sigt = mesh.sigt + t_abs # adding time absorption
        psi = np.zeros((G,N,I))
        for g in range(G):
            for m in range(N):
                psi[g,m,:] += Q[g,m,:]/sigt[g,:]
        return take_moments(psi), psi
    #----------------------------------------------------------------------------------------------
    def invH(Q, geom, t_abs=0.):
        if geom == 'infinite':
            return invH_inf(Q, t_abs=t_abs)
        elif geom == 'slab':
            return invH_slab(Q, t_abs=t_abs)
        elif geom == 'sphere':
            return invH_sphere(Q, t_abs=t_abs)
        else:
            print 'Error: geometry not recognized'
    #----------------------------------------------------------------------------------------------
    def sweep_slab_step(I, dx, mu, sigt, Q, psi_L, psi_R):
        psi_left = np.zeros(I);  psi_right = np.zeros(I)
        if mu > 1e-10:
            psi_left[0] = psi_L
            for i in range(I):
                psi_right[i] = (Q[i] + mu/dx[i]*psi_left[i]) / (mu/dx[i] + sigt[i])
                if i != I-1:
                    psi_left[i+1] = np.copy( psi_right[i] )
        if mu < -1e-10:
            psi_right[I-1] = psi_R
            for i in reversed(range(I)):
                psi_left[i] = (Q[i] - mu/dx[i]*psi_right[i]) / (-mu/dx[i] + sigt[i])
                if i != 0:
                    psi_right[i-1] = np.copy( psi_left[i] )
        psi = 0.5*(psi_left + psi_right)
        return psi, psi_left, psi_right
    #----------------------------------------------------------------------------------------------
    def sweep_slab_diamond_difference(I, dx, mu, sigt, Q, psi_L, psi_R):
        psi_left = np.zeros(I);  psi_right = np.zeros(I)
        if mu > 1e-10:
            psi_left[0] = psi_L
            for i in range(I):
                psi_right[i] = ( Q[i]+(mu/dx[i] - sigt[i]/2.)*psi_left[i] )/( mu/dx[i] + sigt[i]/2. )
                if i != I-1:
                    psi_left[i+1] = np.copy( psi_right[i] )
        if mu < -1e-10:
            psi_right[I-1] = psi_R
            for i in reversed(range(I)):
                psi_left[i] = ( Q[i]+(-mu/dx[i] - sigt[i]/2.)*psi_right[i] )/( -mu/dx[i] + sigt[i]/2. )
                if i != 0:
                    psi_right[i-1] = np.copy( psi_left[i] )
        psi = 0.5*(psi_left + psi_right)
        return psi, psi_left, psi_right     
    #----------------------------------------------------------------------------------------------
    def sweep_starting_direction_diamond_difference(mesh, sigt, g, Q, psi_R):
        I = mesh.num_cells; dx = mesh.dx; sigt = sigt[g,:]        
        psi_left = np.zeros(I);  psi_right = np.zeros(I)
        psi_hat_start = np.zeros(I)

        # for spherical code:
        gamma_hat = mesh.gamma_hat    

        for i in reversed(range(I)):
            if i == I-1:
                psi_right[I-1] = bc.psi_right[g,0]
            else:
                psi_right[i] = psi_left[i+1]
        
            #print psi_right
                
            psi_left[i] = ((1.-sigt[i]*dx[i]/2.)*psi_right[i] + Q[i]*dx[i]) / (1.+sigt[i]*dx[i]/2.)
                  
            psi_hat_start[i] = gamma_hat[i]*psi_right[i] + (1.-gamma_hat[i])*psi_left[i]
        
        return psi_left[0], psi_hat_start 
    #----------------------------------------------------------------------------------------------
    def sweep_sphere_diamond_difference(mesh, g, m, Q, psi_hat_down, psi_L, psi_R):
        I = mesh.num_cells; dx = mesh.dx; mu = mesh.mu[m]; sigt = mesh.sigt[g,:]        
        psi = np.zeros(I); psi_left = np.zeros(I); psi_right = np.zeros(I); 
        psi_hat = np.zeros(I); psi_hat_up = np.zeros(I)

        # for spherical code:
        V = mesh.V
        A_left = mesh.A_left; A_right = mesh.A_right
        gamma =  mesh.gamma; gamma_hat = mesh.gamma_hat      

        w = mesh.w[m]; beta = mesh.beta[m]
        alpha_up = mesh.alpha_up[m]; alpha_down = mesh.alpha_down[m]
        
        if mu < -1e-10:
            psi_right[I-1] = bc.psi_right[g,m]
            for i in reversed(range(I)):
                if i == 0:
                    psi_left[i] = psi_L
                    A_2wb = A_right[i]/(2.*w*beta)
                    ratio = (gamma_hat[i]*psi_right[i]+(1.-gamma_hat[i])*psi_left[i])/(gamma[i]*psi_right[i]+(1.-gamma[i])*psi_left[i])
                    c1 = -mu*A_right[i]*psi_right[i]
                    c2 = (A_2wb*alpha_up*(1.-beta) + A_2wb*alpha_down*beta)*psi_hat_down[i] + Q[i]*V[i]
                    c3 = sigt[i]*V[i] + A_2wb*alpha_up*ratio
                    psi[i] = (c1 + c2)/c3
                    psi_hat[i] = psi[i]*ratio
                    psi_hat_up[i] = 1./beta*(psi_hat[i] - (1.-beta)*psi_hat_down[i])
                    #psi_left[i] = ( Q[i]+(-mu/dx[i] - sigt[i]/2.)*psi_right[i] )/( -mu/dx[i] + sigt[i]/2. )
                else:
                    AA_2wb = (A_right[i] - A_left[i])/(2.*w*beta)
                    c1 = 1./(-mu*A_left[i] + AA_2wb*alpha_up*(1.-gamma_hat[i]) + sigt[i]*V[i]*(1.-gamma[i]) )
                    c2 = (-mu*A_right[i] - AA_2wb*alpha_up*gamma_hat[i] - sigt[i]*V[i]*gamma[i])*psi_right[i]
                    c3 = (AA_2wb*alpha_up*(1.-beta) + AA_2wb*alpha_down*beta)*psi_hat_down[i] + Q[i]*V[i]
                    psi_left[i] = c1*(c2 + c3)
                    psi[i] = gamma[i]*psi_right[i] + (1.-gamma[i])*psi_left[i]
                    psi_hat[i] = gamma_hat[i]*psi_right[i] + (1.-gamma_hat[i])*psi_left[i]
                    psi_hat_up[i] = 1./beta*(psi_hat[i] - (1.-beta)*psi_hat_down[i])
                    #print c1, c2, c3, psi_hat_down
                    #psi_left[i] = ( Q[i]+(-mu/dx[i] - sigt[i]/2.)*psi_right[i] )/( -mu/dx[i] + sigt[i]/2. )
                    psi_right[i-1] = np.copy( psi_left[i] )
        if mu > 1e-10:
            psi_left[0] = psi_L
            for i in range(I):
                AA_2wb = (A_right[i] - A_left[i])/(2.*w*beta)
                c1 = 1./(mu*A_right[i] + AA_2wb*alpha_up*gamma_hat[i] + sigt[i]*V[i]*gamma[i])
                c2 = (mu*A_left[i] - AA_2wb*alpha_up*(1.-gamma_hat[i]) - sigt[i]*V[i]*(1.-gamma[i]))*psi_left[i]
                c3 = (AA_2wb*alpha_up*(1.-beta) + AA_2wb*alpha_down*beta)*psi_hat_down[i] + Q[i]*V[i]
                psi_right[i] = c1*(c2 + c3)
                psi[i] = gamma[i]*psi_right[i] + (1.-gamma[i])*psi_left[i]
                psi_hat[i] = gamma_hat[i]*psi_right[i] + (1.-gamma_hat[i])*psi_left[i]
                psi_hat_up[i] = 1./beta*(psi_hat[i] - (1.-beta)*psi_hat_down[i])
                #psi_right[i] = ( Q[i]+(mu/dx[i] - sigt[i]/2.)*psi_left[i] )/( mu/dx[i] + sigt[i]/2. )
                if i != I-1:
                    psi_left[i+1] = np.copy( psi_right[i] )

        return psi, psi_left, psi_right, psi_hat_up      
    #----------------------------------------------------------------------------------------------
    def S(phi):
        S = np.zeros((G,N,I))
        for i in range(I):
            p = mesh.mat_p[mesh.cell_mat[i]]
            for moment in range(L):
                for g in range(G):
                    sum_over_gp = 0.
                    for gp in range(G):
                        sum_over_gp += p[moment][gp,g]*phi[gp,moment,i]
                    for m in range(N):
                        S[g,m,i] += mesh.legendre_and_normalization[moment,m]*sum_over_gp
        return S
    #----------------------------------------------------------------------------------------------
    def F(phi):
        F = np.zeros((G,N,I))
        for i in range(I):
            sigf = mesh.mat_sigf[mesh.cell_mat[i]]
            for g in range(G):
                sum_over_gp = 0
                for gp in range(G):
                    sum_over_gp += sigf[gp,g]*phi[gp,0,i] 
                for m in range(N):
                    F[g,m,i] += 0.5*sum_over_gp
        return F
    #----------------------------------------------------------------------------------------------
    def DSA(phi, phi_old, k):    
        phi_new = np.copy(phi)

        for g in range(G):
            A = np.matrix(np.zeros((I,I)))
            b = np.matrix(np.zeros((I,1)))
                
            A[0,0] = 1.
            A[I-1,I-1] = 1.
            b[0,0] = 0.
            b[I-1,0] = 0.

            for i in range(1,I-1):
                D = mesh.mat_D[mesh.cell_mat[i]][g]
                siga = mesh.mat_siga[mesh.cell_mat[i]][g]
                p = mesh.mat_p[mesh.cell_mat[i]] 
                f = mesh.mat_sigf[mesh.cell_mat[i]]/k

                A[i,i-1] = -D/(2.*mesh.dx[i])
                A[i,i] = 2.*D/(2.*mesh.dx[i]) + siga
                A[i,i+1] = -D/(2.*mesh.dx[i])              

                for gp in range(G):
                    b[i,0] += (p[0][gp,g]+f[gp,g])*(phi[gp,0,i] - phi_old[gp,0,i])

            d_phi = linalg.gmres(A,b,tol=1e-14)[0]
            
            for i in range(I):
                phi_new[g,0,i] += d_phi[i] 
            
        return phi_new
    #----------------------------------------------------------------------------------------------
    if discretization == 'cs':
        mapping = mesh.mapping
        E = mesh.num_c_grps
        print "mapping ", mapping

        #------------------------------------------------------------------------------------------
        def coarse(phi):
            coarse_phi = np.zeros((E,L,I))
            for e in range(E):  
                for g in mapping[e]:  
                    for moment in range(L): 
                        for i in range(I):
                            coarse_phi[e,moment,i] += phi[g,moment,i]
            return coarse_phi
        #------------------------------------------------------------------------------------------
        def recompute_S_eg(moment, e, phi, coarse_phi):
            #if e == 0:
            #    print "Recomputing S_e->g for moment:  ", moment
            S_moment_e = np.zeros((G,I))
            for i in range(I):
                p = mesh.mat_p[mesh.cell_mat[i]]
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for gp in range(G):   
                        numer += p[moment][gp,g]*phi[gp,moment,i]#+1e-16
                    for ep in range(E):
                        denom += coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]#+1e-16
                    numer += 1e-16
                    denom += 1e-16
                    S_moment_e[g,i] = numer/denom
            #if talk==True: print "S_eg was recomputed"
            return S_moment_e
        #------------------------------------------------------------------------------------------
        def recompute_F_eg(e, phi, coarse_phi):
            #if e == 0:
            #    print "Recomputing F_e->g"
            F_e = np.zeros((G,I))
            for i in range(I):
                sigf = mesh.mat_sigf[mesh.cell_mat[i]]
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for gp in range(G):   
                        numer += sigf[gp,g]*phi[gp,0,i]#+1e-16
                    for ep in range(E):
                        denom += coarse_sigf[ep,e]*coarse_phi[ep,0,i]#+1e-16 
                    numer += 1e-16
                    denom += 1e-16               
                    F_e[g,i] = numer/denom
            #if talk==True: print "F_eg was recomputed"
            return F_e
        #------------------------------------------------------------------------------------------
        def CS(coarse_phi, S_eg):
            CS = np.zeros((G,N,I))
            for i in range(I):
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for moment in range(L):
                    for e in range(E):
                        for g in mapping[e]: 
                            sum_over_ep = 0
                            for ep in range(E):
                                sum_over_ep += S_eg[moment,e,g,i]*coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]
                            for m in range(N):
                                CS[g,m,i] += mesh.legendre_and_normalization[moment,m]*sum_over_ep
            return CS
        #------------------------------------------------------------------------------------------
        def CF(coarse_phi, F_eg):
            CF = np.zeros((G,N,I))
            for i in range(I):
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                for e in range(E):
                    for g in mapping[e]:
                        sum_over_ep = 0 
                        for ep in range(E):
                            sum_over_ep += F_eg[e,g,i]*coarse_sigf[ep,e]*coarse_phi[ep,0,i]
                        for m in range(N):
                            CF[g,m,i] += 0.5*sum_over_ep
            return CF
    #----------------------------------------------------------------------------------------------
    def tot_source(Q_ext):
        tot_source = 0                   
        for i in range(I):
            for g in range(G):
                for m in range(N):
                    tot_source += Q_ext[g,m,i]*mesh.w[m]*mesh.V[i]
        return tot_source
    #----------------------------------------------------------------------------------------------
    def tot_rxn(phi):
        tot_rxn = 0
        for g in range(G):
            for i in range(I):
                tot_rxn += mesh.sigt[g,i]*phi[g,0,i]*mesh.V[i]
        return tot_rxn
    #----------------------------------------------------------------------------------------------
    def leak_left(psi_left):
        leak_left = 0
        for g in range(G):
            for m in range(N):
                leak_left -= psi_left[g,m,0]*mesh.mu[m]*mesh.w[m]
        leak_left *= mesh.A_left[0]
        return leak_left
    #----------------------------------------------------------------------------------------------
    def leak_right(psi_right):
        leak_right = 0
        for g in range(G):
            for m in range(N):
                leak_right += psi_right[g,m,I-1]*mesh.mu[m]*mesh.w[m]
        leak_right *= mesh.A_right[I-1]
        return leak_right
    #----------------------------------------------------------------------------------------------
    def residual(phi, coarse_phi, S_eg, F_eg=None):
        residual = np.zeros(G)
        S_phi = S(phi)[0]
        CS_phi = CS(coarse_phi, S_eg)[0]
        if F_eg != None:
            F_phi = F(phi)[0]
            CF_phi = CF(coarse_phi, F_eg)[0]
        for i in range(I):
            for m in range(N):
                for g in range(G):
                    residual[g] += S_phi[g,m,i] - CS_phi[g,m,i]
                    if F_eg != None: 
                        residual[g] += F_phi[g,m,i] - CF_phi[g,m,i]
        return abs(sum(residual))
    #----------------------------------------------------------------------------------------------
    def determine_balance(Q, phi, psi_left, psi_right):
        A = tot_source(Q)
        B = tot_rxn(phi)
        C = leak_left(psi_left)
        D = leak_right(psi_right)
        balance = (A - B - C - D)/A

        print("------------------------------") 
        print("total source = %.2e" %A)
        print("total reaction rate = %.2e" %B)
        print("left-side leakage = %.2e" %C) 
        print("right-side leakage = %.2e" %D) 
        print("------------------------------") 
        print("balance = %.2e" %balance) 
        print("------------------------------") 
    #----------------------------------------------------------------------------------------------
    def source_mg(phys, Q_ext):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi

        Q = S(phi) + F(phi) + Q_ext
        phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom)
        #determine_balance(Q, phi_new, psi_left_new, psi_right_new)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 

        return phys_new
    #----------------------------------------------------------------------------------------------
    def source_cs(phys, Q_ext, F_eg, S_eg):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi   

        c_phi = coarse(phi);   

        Q = CS(c_phi,S_eg) + CF(c_phi,F_eg) + Q_ext
        phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom)
        #determine_balance(Q, phi_new, psi_left_new, psi_right_new)

        c_phi_new = coarse(phi_new)

        for e in range(E):  
            if iter%recomp_F==0:
                F_eg[e] = recompute_F_eg(e, phi_new, c_phi_new)

        for moment in range(L):     
            for e in range(E):  
                if iter%recomp_S[moment] == 0:
                    S_eg[moment,e] = recompute_S_eg(moment, e, phi_new, c_phi_new)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 

        return phys_new, F_eg, S_eg
    #----------------------------------------------------------------------------------------------
    def k_mg(phys, method='power'):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi; k = phys.k 

        if method == 'rayleigh':
            phi_S, psi_S, psi_left_S, psi_right_S = invH(S(phi), geom)
            phi_F, psi_F, psi_left_F, psi_right_F = invH(F(phi), geom)
            #determine_balance(S(phi)+F(phi), phi_S+phi_F, psi_left_S+psi_left_F, psi_right_S++psi_right_F)

            k_new = inner_prod(psi, psi_F)/ (inner_prod(psi, psi) - inner_prod(psi, psi_S))

            phi_new = phi_S + phi_F/k_new
            psi_new = psi_S + psi_F/k_new

        elif method == 'power':
            Q = S(phi) + F(phi)/k
            phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom)
            #determine_balance(Q, phi_new, psi_left_new, psi_right_new)

            k_new = k*np.linalg.norm(F(phi_new)*mesh.V)/np.linalg.norm(F(phi)*mesh.V) 

            phi_new /= np.linalg.norm(phi_new)
            psi_new /= np.linalg.norm(psi_new)

        else:
            print("ERROR: '%s' method not available" %method)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 
        phys_new.k = np.copy(k_new);

        return phys_new
    #----------------------------------------------------------------------------------------------
    def k_cs(phys, F_eg, S_eg, method='power'):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi; k = phys.k 

        if method == 'rayleigh':
            c_phi = coarse(phi);   

            for e in range(E):  
                if iter%recomp_F==0:
                    F_eg[e] = recompute_F_eg(e, phi, c_phi)
            CF_phi = CF(c_phi,F_eg)
            
            for moment in range(L):     
                for e in range(E):  
                    if iter%recomp_S[moment] == 0:
                        S_eg[moment,e] = recompute_S_eg(moment, e, phi, c_phi)
            CS_phi = CS(c_phi,S_eg)

            phi_F, psi_F, psi_left_F, psi_right_F = invH(CF_phi, geom)
            phi_S, psi_S, psi_left_S, psi_right_S = invH(CS_phi, geom)

            k_new = inner_prod(psi, psi_F)/ (inner_prod(psi, psi) - inner_prod(psi, psi_S))

            phi_new = phi_S + phi_F/k_new
            psi_new = psi_S + psi_F/k_new

        elif method == 'power':
            c_phi = coarse(phi);   

            CS_phi = CS(c_phi,S_eg)
            CF_phi = CF(c_phi,F_eg)

            Q = CS_phi + CF_phi/k
            phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom)
            determine_balance(S(phi)+F(phi)/k, phi_new, psi_left_new, psi_right_new)

            c_phi_new = coarse(phi_new)

            for e in range(E):  
                if iter%recomp_F==0:
                    F_eg[e] = recompute_F_eg(e, phi_new, c_phi_new)

            for moment in range(L):     
                for e in range(E):  
                    if iter%recomp_S[moment] == 0:
                        S_eg[moment,e] = recompute_S_eg(moment, e, phi_new, c_phi_new)

            CF_phi_new = CF(c_phi_new,F_eg)
             
            k_new = k*np.linalg.norm(CF_phi_new*mesh.V)/np.linalg.norm(CF_phi*mesh.V)  
            #k_new = k*np.linalg.norm(F(phi_new)*mesh.V)/np.linalg.norm(F(phi)*mesh.V) 

            phi_new /= np.linalg.norm(phi_new)
            psi_new /= np.linalg.norm(phi_new)

        else:
            print("ERROR: '%s' method not available" %method)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 
        phys_new.k = np.copy(k_new);

        return phys_new, F_eg, S_eg
    #----------------------------------------------------------------------------------------------
    def alpha_mg(phys, min_alpha=-1e10, max_alpha=1e10, method='rayleigh'):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi; alpha = phys.alpha

        if method == 'rayleigh':
            Q = S(phi)+F(phi)

            phi_Q, psi_Q, psi_left_Q, psi_right_Q = invH(Q, geom)
            phi_V, psi_V, psi_left_V, psi_right_V = invH(invV(N)*psi, geom)

            alpha = (inner_prod(psi, psi_Q) - inner_prod(psi, psi)) / inner_prod(psi, psi_V)
            alpha = max(min(alpha, max_alpha), min_alpha)

            phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom, t_abs=alpha*invV())

        else:
            print("ERROR: '%s' method not available" %method)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 
        phys_new.alpha = np.copy(alpha);

        return phys_new
    #----------------------------------------------------------------------------------------------
    def alpha_cs(phys, F_eg, S_eg, min_alpha=-1e10, max_alpha=1e10, method='rayleigh'):
        phys_new = copy.copy(phys); phi = phys.phi; psi = phys.psi; alpha = phys.alpha

        if method == 'rayleigh':
            c_phi = coarse(phi);   

            for e in range(E):  
                if iter%recomp_F==0:
                    F_eg[e] = recompute_F_eg(e, phi, c_phi)
            CF_phi = CF(c_phi,F_eg)
            
            for moment in range(L):     
                for e in range(E):  
                    if iter%recomp_S[moment] == 0:
                        S_eg[moment,e] = recompute_S_eg(moment, e, phi, c_phi)
            CS_phi = CS(c_phi,S_eg)

            Q = CS_phi + CF_phi

            phi_Q, psi_Q, psi_left_Q, psi_right_Q = invH(Q, geom)
            phi_V, psi_V, psi_left_V, psi_right_V = invH(invV(N)*psi, geom)

            alpha = (inner_prod(psi, psi_Q) - inner_prod(psi, psi)) / inner_prod(psi, psi_V)
            alpha = max(min(alpha, max_alpha), min_alpha)

            phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom, t_abs=alpha*invV())

        else:
            print("ERROR: '%s' method not available" %method)

        phys_new.phi = np.copy(phi_new); phys_new.psi = np.copy(psi_new); 
        phys_new.alpha = np.copy(alpha);

        return phys_new, F_eg, S_eg
    #----------------------------------------------------------------------------------------------   
    t = time.time();  t0 = time.time()
    phys = Physics(mesh, L_max=L_max); phys_new = Physics(mesh, L_max=L_max); phi_error = 1.;   
    iter = 0; iteration_dict = []; geom = mesh.geom; B = 0; rho = 0; # spectral radius

    if discretization == 'cs':
        S_eg = np.zeros((L,E,G,I))
        F_eg = np.zeros((E,G,I))
        for e in range(E):
            for moment in range(L):
                S_eg[moment, e] = recompute_S_eg(moment, e, phys_new.phi, phys_new.c_phi)
            F_eg[e] = recompute_F_eg(e, phys_new.phi, phys_new.c_phi)
    if problem_type == "source":
        print "\nCommencing source iterations:"

        Q_ext = np.ones((G,N,I))
        #Q_ext = np.ones((G,N,I))*mesh.dx       

        while phi_error > tol*(1-rho) and iter < max_its+1 or (iter-1)%4 != 0: 
            phys_old = copy.copy(phys)                
            phys     = copy.copy(phys_new)

            if discretization == "mg":
                phys_new = source_mg(phys, Q_ext)
                                                                                      
            elif discretization == "cs":
                phys_new, F_eg, S_eg = source_cs(phys, Q_ext, F_eg, S_eg)
                            
            phi_error = np.linalg.norm(phys_new.phi - phys.phi)/np.linalg.norm(phys.phi)

            if iter > 1:
                rho = np.linalg.norm( phys_new.phi - phys.phi ) / np.linalg.norm( phys.phi - phys_old.phi )
            #print rho

            iter+=1

            runtime = time.time()-t
            print ("iter %3i  phi_error = %.1e  tot_runtime = %.1f\n" %(iter,phi_error,runtime) )
                                                                                
            
            if L_max > 1:
                iteration_dict.append({'runtime': runtime, 'phi_error': np.copy(phi_error),
                                        'flux':np.copy(phys_new.phi[:,0,:]), 'current':np.copy(phys_new.phi[:,1,:])})
            else:
                iteration_dict.append({'runtime': runtime, 'phi_error': np.copy(phi_error), 'flux':np.copy(phys_new.phi[:,0,:])})        
                
        runtime = time.time() - t
        return phys_new.phi, phys_new.psi/np.linalg.norm(phys_new.psi), runtime, iteration_dict

    if problem_type == "k":
        print "\nCommencing Power iterations to calculate k:"
        k_error = 1.; V = mesh.V; phys_F = Physics(mesh); phys_S = Physics(mesh)
        while (k_error > tol*(1-rho) or phi_error > tol*(1-rho)) and iter < max_its+1 or (iter-1)%4 != 0: 
            phys_old = copy.copy(phys)                
            phys     = copy.copy(phys_new)    

            if discretization == "mg":
                phys_new = k_mg(phys)
                                                                                        
            elif discretization == "cs":
                phys_new, F_eg, S_eg = k_cs(phys, F_eg, S_eg)

            if DSA_opt == True:
                phys_new.phi = DSA(phys_new.phi, phys.phi, phys_new.k)

            phi_error = np.linalg.norm(phys_new.phi - phys.phi)

            if k_exact == None:
	            k_error = np.abs(phys_new.k - phys.k)/phys.k
            else:
            	k_error = abs(phys_new.k - k_exact)/k_exact 

            if iter > 1:
                rho = np.linalg.norm( phys_new.phi - phys.phi ) / np.linalg.norm( phys.phi - phys_old.phi )
            rho = 0
            #print rho

            iter += 1

            runtime = time.time()-t
            print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  tot_runtime = %.1fs\n" %(iter,phys_new.k,k_error,phi_error,runtime) )
                                        
            if L_max > 1:
                iteration_dict.append({'runtime': runtime, 'k': np.copy(phys_new.k), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error), 'flux':np.copy(phys_new.phi[:,0,:]), 'current':np.copy(phys_new.phi[:,1,:])})
            else:
                iteration_dict.append({'runtime': runtime, 'k': np.copy(phys_new.k), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error), 'residual': np.copy(B), 'flux':np.copy(phys_new.phi[:,0,:])})        
                
        runtime = time.time() - t
        return phys_new.k, phys_new.phi, phys_new.psi/np.linalg.norm(phys_new.psi), runtime, iteration_dict

    elif problem_type == "alpha":
        alpha = 0; old_alpha = 0; d_alpha = 1.; iter=0; psi_new = np.ones((G,N,I))

        min_alpha = -1e10 
        max_alpha =  1e10     
        for i in range(I):
            for g in range(G):
                min_alpha = max(min_alpha,-invV()[g,i]/mesh.sigt[g,i]*(1-1e-7))
                print min_alpha        

        while (abs(d_alpha) > tol*(1-rho) or phi_error > tol*(1-rho)) and iter < max_its:  
            phys_old = copy.copy(phys)                
            phys     = copy.copy(phys_new)    

            if discretization == "mg":
                phys_new = alpha_mg(phys, min_alpha=min_alpha, max_alpha=max_alpha)
                                                                                        
            elif discretization == "cs":
                phys_new, F_eg, S_eg = alpha_cs(phys, F_eg, S_eg, min_alpha=min_alpha, max_alpha=max_alpha)

            phi_error = np.linalg.norm(phys_new.phi/np.linalg.norm(phys_new.phi) - phys.phi/np.linalg.norm(phys.phi))
            d_alpha = phys_new.alpha - phys.alpha

            iter += 1
            if talk==True and iter%1 == 0: print ("iteration %3i  alpha = %.10e  d_alpha = %.3e  phi_error = %.3e" %(iter,phys_new.alpha,d_alpha,phi_error) )

        runtime = time.time() - t
        return alpha, phys_new.phi, phys_new.psi/np.linalg.norm(phys_new.psi), runtime
###################################################################################################

