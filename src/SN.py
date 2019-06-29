import numpy as np
import time
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
            self.mat_chid[mat] = mat_dict[mat].pdt_chid
            self.mat_beta_fnubar[mat] = mat_dict[mat].pdt_beta_fnubar
            if '18' in mat_dict[mat].crossSectionMTList: 
                self.mat_sigf_MT18[mat] = mat_dict[mat].sigf_MT18
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
                    inner_prod += x[g,m,i]*Ax[g,m,i]
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
        psi = np.zeros((G,N,I)); psi_left = np.zeros((G,N,I)); psi_right = np.zeros((G,N,I)); count=0
        sigt = mesh.sigt + t_abs # adding time absorption
        for g in range(G):
            for m in range(N):
                psi_L = 0.
                psi_R = 0.
                psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:] = sweep_slab_diamond_difference(I, mesh.dx, mesh.mu[m], mesh.sigt[g,:], Q[g,m,:], psi_L, psi_R)
                count += I #15*I
        return take_moments(psi), psi, psi_left, psi_right, count
    #----------------------------------------------------------------------------------------------
    def invH_sphere(Q, t_abs=0.):
        # sweeps across all cells for all energies and directions
        psi = np.zeros((G,N,I)); psi_left = np.zeros((G,N,I)); psi_right = np.zeros((G,N,I));
        psi_hat_down = np.zeros((G,N,I)); psi_hat_up = np.zeros((G,N,I)); count=0
        mu = mesh.mu
        psi_L = np.zeros(G); psi_R = np.zeros(G)
        sigt = mesh.sigt + t_abs # adding time absorption
        for g in range(G):
            psi_R[g] = 0.
            for m in range(N):
                if m == 0:
                    Q_start = Q[g,0,:]*(mu[1]+1.)/(mu[1]-mu[0]) - Q[g,1,:]*(1.+mu[0])/(mu[1]-mu[0])
                    psi_L[g], psi_hat_down[g,0,:] = sweep_starting_direction_diamond_difference(mesh, g, Q_start, psi_R[g])
                else:
                    psi_hat_down[g,m,:] = np.copy(psi_hat_up[g,m-1,:])
                psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:], psi_hat_up[g,m,:] = sweep_sphere_diamond_difference(mesh, g, m, Q[g,m,:], psi_hat_down[g,m,:], psi_L[g], psi_R[g])
                count += I #15*I

        return take_moments(psi), psi, psi_left, psi_right, count
    #----------------------------------------------------------------------------------------------
    def invH_inf(Q, t_abs=0.):
        # infinite-medium Boltzmann operator (no streaming)
        sigt = mesh.sigt + t_abs # adding time absorption
        psi = np.zeros((G,N,I)); count=0
        for g in range(G):
            for m in range(N):
                psi[g,m,:] += Q[g,m,:]/sigt[g,:]
                count += 1
        return take_moments(psi), psi, count  
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
    def sweep_starting_direction_diamond_difference(mesh, g, Q, psi_R):
        I = mesh.num_cells; dx = mesh.dx; sigt = mesh.sigt[g,:]        
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
        S = np.zeros((G,N,I)); count=0
        for i in range(I):
            p = mesh.mat_p[mesh.cell_mat[i]]
            for moment in range(L):
                for g in range(G):
                    sum_over_gp = 0.
                    for gp in range(G):
                        sum_over_gp += p[moment][gp,g]*phi[gp,moment,i]
                        count += 1
                    for m in range(N):
                        S[g,m,i] += mesh.legendre_and_normalization[moment,m]*sum_over_gp
                        count += 1
        return S, count
    #----------------------------------------------------------------------------------------------
    def F(phi):
        F = np.zeros((G,N,I)); count = 0
        for i in range(I):
            sigf = mesh.mat_sigf[mesh.cell_mat[i]]
            for g in range(G):
                sum_over_gp = 0
                for gp in range(G):
                    sum_over_gp += sigf[gp,g]*phi[gp,0,i] 
                    count += 1
                for m in range(N):
                    F[g,m,i] += 0.5*sum_over_gp
                    count += 1
        return F, count
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
            S_moment_e = np.zeros((G,I)); count=0
            for i in range(I):
                p = mesh.mat_p[mesh.cell_mat[i]]
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for gp in range(G):   
                        numer += p[moment][gp,g]*phi[gp,moment,i]#+1e-16
                        count += 1
                    for ep in range(E):
                        denom += coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]#+1e-16
                        count += 1
                    numer += 1e-16
                    denom += 1e-16
                    S_moment_e[g,i] = numer/denom
            #if talk==True: print "S_eg was recomputed"
            return S_moment_e, count
        #------------------------------------------------------------------------------------------
        def recompute_F_eg(e, phi, coarse_phi):
            #if e == 0:
            #    print "Recomputing F_e->g"
            F_e = np.zeros((G,I)); count=0
            for i in range(I):
                sigf = mesh.mat_sigf[mesh.cell_mat[i]]
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for gp in range(G):   
                        numer += sigf[gp,g]*phi[gp,0,i]#+1e-16
                        count += 1
                    for ep in range(E):
                        denom += coarse_sigf[ep,e]*coarse_phi[ep,0,i]#+1e-16
                        count += 1     
                    numer += 1e-16
                    denom += 1e-16               
                    F_e[g,i] = numer/denom
            #if talk==True: print "F_eg was recomputed"
            return F_e, count
        #------------------------------------------------------------------------------------------
        def CS(coarse_phi, S_eg):
            CS = np.zeros((G,N,I)); count=0
            for i in range(I):
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for moment in range(L):
                    for e in range(E):
                        for g in mapping[e]: 
                            sum_over_ep = 0
                            for ep in range(E):
                                sum_over_ep += S_eg[moment,e,g,i]*coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]
                                count += 1  
                            for m in range(N):
                                CS[g,m,i] += mesh.legendre_and_normalization[moment,m]*sum_over_ep
                                count += 1
            return CS, count
        #------------------------------------------------------------------------------------------
        def CF(coarse_phi, F_eg):
            CF = np.zeros((G,N,I)); count=0
            for i in range(I):
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                for e in range(E):
                    for g in mapping[e]:
                        sum_over_ep = 0 
                        for ep in range(E):
                            sum_over_ep += F_eg[e,g,i]*coarse_sigf[ep,e]*coarse_phi[ep,0,i]
                            count += 1
                        for m in range(N):
                            CF[g,m,i] += 0.5*sum_over_ep
                            count += 1
            return CF, count
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
    t = time.time()
    phi = np.ones((G,L,I))
    phi_new = np.ones((G,L,I))
    phi_error = 1.;  iter = 0; t0 = time.time(); iteration_dict = []    
    rho = 0; # spectral radius
    B = 0;
    geom = mesh.geom;

    if problem_type == "source":
        for mat in sorted(mesh.mat_dict.keys()):
            for moment in range(N):
                mesh.mat_p[mat][0] *= 0.
                mesh.mat_coarse_p[mat][0] *= 0.
            #mesh.mat_sigf[mat] *= 0.
            #mesh.mat_coarse_sigf[mat] *= 0.

        for mat in sorted(mesh.mat_dict.keys()):
            mesh.mat_p[mat][0] += mesh.mat_sigf[mat]
            mesh.mat_coarse_p[mat][0] += mesh.mat_coarse_sigf[mat]

    if discretization == 'cs':
        c_phi_new = coarse(phi_new)
        mapping = mesh.mapping
        S_eg_new = np.zeros((L,E,G,I))
        F_eg_new = np.zeros((E,G,I))
        for e in range(E):
            for moment in range(L):
                S_eg_new[moment, e], S_eg_count = recompute_S_eg(moment, e, phi_new, c_phi_new);      S_eg_tot_count = S_eg_count
            F_eg_new[e], F_eg_count = recompute_F_eg(e, phi_new, c_phi_new);                          F_eg_tot_count = F_eg_count 
        
        phi_ratio_at_prev_S_eg=np.zeros(G)
        phi_ratio_at_prev_F_eg=np.zeros(G)

    if problem_type == "source":
        print "\nCommencing source iterations:"
        tot_figure_of_merit=0; LHS_cost = 0; RHS_cost = 0;

        #Q_ext = np.ones((G,N,I))*mesh.dx
        A = mesh.x_edges[-1] - mesh.x_edges[0]

        Q_ext = np.zeros((G,N,I))
        '''
        for g in range(G):
            for m in range(N):
                for i in range(I):
                    Q_ext[g,m,i] = mesh.mu[m]*np.pi/A*np.cos(np.pi*mesh.x_mids[i]/A)+np.sin(np.pi*mesh.x_mids[i]/A)

                    #Q_ext[g,m,i] = mesh.mu[m]*np.pi/A*np.cos(np.pi*mesh.x_mids[i]/A)+(mesh.sigt[0,0]-mesh.mat_p[sorted(mesh.mat_dict.keys())[0]][0][0,0])*np.sin(np.pi*mesh.x_mids[i]/A)
        '''

        while phi_error > tol*(1-rho) and iter < max_its+1 or (iter-1)%4 != 0: 
            phi_old = np.copy(phi)                
            phi     = np.copy(phi_new)

            if discretization == "mg":
                S_tot_count = 0; H_tot_count = 0

                S_phi, S_count = S(phi);                        S_tot_count += S_count; print np.linalg.norm(S_phi.flatten())
                Q = S_phi + Q_ext;                              t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new, H_count = invH(Q, geom);    H_tot_count += H_count;  t_H = (time.time() - t0) - t_Q

                if mode=='debug':
                    #print "psi", psi_new
                    #print 
                    #print "phi", phi_new
                    print tot_source(Q)
                    print tot_rxn(phi_new)
                    print leak_left(psi_left_new)
                    print leak_right(psi_right_new)
                    balance = tot_source(Q) - tot_rxn(phi_new) - leak_left(psi_left_new) - leak_right(psi_right_new)
                    print "balance = ", np.linalg.norm(balance) 
                
                phi_error = np.linalg.norm(phi_new - phi)

                if iter > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                #print rho

                iter += 1

                figure_of_merit = H_tot_count + S_tot_count 
                tot_figure_of_merit += figure_of_merit
                LHS_cost += H_tot_count
                RHS_cost += S_tot_count 
                runtime = time.time()-t
                FOM_RT= tot_figure_of_merit/runtime
                print ("iter %3i  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" 
                                                                        %(iter,phi_error,t_Q,t_H,runtime) )
                print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  S_count = %.2e\n\n" %(tot_figure_of_merit,RHS_cost, H_tot_count, S_tot_count))

                                                                                        
            elif discretization == "cs":
                CS_tot_count = 0; H_tot_count = 0; S_eg_tot_count = 0

                c_phi = np.copy(c_phi_new) 
                S_eg  = np.copy(S_eg_new)    

                CS_phi, CS_count = CS(c_phi,S_eg);                CS_tot_count += CS_count; print np.linalg.norm(CS_phi.flatten())
                Q = CS_phi + Q_ext;                               t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new, H_count = invH(Q, geom);              H_tot_count += H_count; 
                c_phi_new = coarse(phi_new);                      t_H = (time.time() - t0) - t_Q
 
                for moment in range(L):     
                    for e in range(E):  
                        if iter%recomp[moment] == 0:
                            S_eg_new[moment,e], S_eg_count = recompute_S_eg(moment, e, phi_new, c_phi_new);      S_eg_tot_count += S_eg_count

                if mode=='debug':
                    #print tot_source(phi_new, Q_ext)
                    #print tot_rxn(phi_new)
                    #print leak_left(psi_left_new)
                    #print leak_right(psi_right_new)
                    #balance = tot_source(phi_new, Q_ext) - tot_rxn(phi_new) - leak_left(psi_left_new) - leak_right(psi_right_new)
                    #print "balance = ", np.linalg.norm(balance)

                    B = residual(phi_new, c_phi_new, S_eg_new) 
                    print "residual = ", B
                    
                            
                phi_error = np.linalg.norm(phi_new - phi)/np.linalg.norm(phi)

                if iter > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                #print rho

                iter+=1

                figure_of_merit = H_tot_count + CS_tot_count + S_eg_tot_count
                tot_figure_of_merit += figure_of_merit
                LHS_cost += H_tot_count
                RHS_cost += CS_tot_count + S_eg_tot_count 
                runtime = time.time()-t
                FOM_RT= tot_figure_of_merit/runtime
                print ("iter %3i  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" 
                                                                        %(iter,phi_error,t_Q,t_H,runtime) )
                print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  CS_count = %.2e  S_eg_tot_count = %.2e \n\n"                                                     %(tot_figure_of_merit, RHS_cost, H_tot_count, CS_tot_count, S_eg_tot_count) )
                                                                                    
            
            if L_max > 1:
                iteration_dict.append({'runtime': runtime, 'phi_error': np.copy(phi_error),
                                        'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                        'flux':np.copy(phi_new[:,0,:]), 'current':np.copy(phi_new[:,1,:])})
            else:
                iteration_dict.append({'runtime': runtime, 'phi_error': np.copy(phi_error),
                                        'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                        'flux':np.copy(phi_new[:,0,:])})        
                
            t0 = time.time()
        runtime = time.time() - t
        return phi_new, psi_new/np.linalg.norm(psi_new), runtime, iteration_dict

 
    elif problem_type == "k":
        print "\nCommencing Power iterations to calculate k:"
        k_new = 1.; k_error = 1.; tot_figure_of_merit=0; LHS_cost = 0; RHS_cost = 0;
        while (k_error > tol*(1-rho) or phi_error > tol*(1-rho)) and iter < max_its+1 or (iter-1)%4 != 0: 
            phi_old = np.copy(phi)                
            phi     = np.copy(phi_new)
            k       = np.copy(k_new)

            if discretization == "mg":
                S_tot_count = 0; F_tot_count = 0; H_tot_count = 0

                S_phi, S_count = S(phi);                        S_tot_count += S_count; 
                F_phi, F_count = F(phi);                        F_tot_count += F_count;
                Q = S_phi + F_phi/k;                            t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new, H_count = invH(Q, geom);  H_tot_count += H_count;  t_H = (time.time() - t0) - t_Q

                if mode=='debug':
                    print tot_source(Q)
                    print tot_rxn(phi_new)
                    print leak_left(psi_left_new)
                    print leak_right(psi_right_new)
                    balance = tot_source(Q) - tot_rxn(phi_new) - leak_left(psi_left_new) - leak_right(psi_right_new)
                    print "balance = ", balance
                
                k_new   = k*np.linalg.norm(F(phi_new)[0])/np.linalg.norm(F(phi)[0]) 
                #k_new = k*inner_prod(phi_new, F(phi_new))/ inner_prod(phi_new, (H() - S(phi_new)))

                if DSA_opt == True:
                    phi_new = DSA(phi_new,phi,k_new)

                phi_new   /= np.linalg.norm(phi_new)
                phi_error = np.linalg.norm(phi_new - phi)

                if k_exact == None:
		            k_error = np.abs(k_new - k)/k
                else:
                	k_error = abs(k_new - k_exact)/k_exact 

                if iter > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                rho = 0
                print rho

                iter += 1

                figure_of_merit = H_tot_count + S_tot_count + F_tot_count
                tot_figure_of_merit += figure_of_merit
                LHS_cost += H_tot_count
                RHS_cost += S_tot_count + F_tot_count 
                runtime = time.time()-t
                FOM_RT= tot_figure_of_merit/runtime
                print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" %(iter,k,k_error,phi_error,t_Q,t_H,runtime) )
                print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  S_count = %.2e  F_count = %.2e\n\n" %(tot_figure_of_merit,RHS_cost, H_tot_count, S_tot_count, F_tot_count))
                                                                                        
            elif discretization == "cs":
                CS_tot_count = 0; CF_tot_count = 0; H_tot_count = 0; S_eg_tot_count = 0; F_eg_tot_count = 0

                c_phi = np.copy(c_phi_new)                
                S_eg  = np.copy(S_eg_new)  
                F_eg  = np.copy(F_eg_new)    

                CS_phi, CS_count = CS(c_phi,S_eg);                CS_tot_count += CS_count;
                CF_phi, CF_count = CF(c_phi,F_eg);                CF_tot_count += CF_count;
                Q = CS_phi + CF_phi/k;                            t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new, H_count = invH(Q, geom); H_tot_count += H_count; 
                c_phi_new = coarse(phi_new);                      t_H = (time.time() - t0) - t_Q
 
                for e in range(E):  
                    if iter%recomp_F==0:
                        F_eg_new[e], F_eg_count = recompute_F_eg(e, phi_new, c_phi_new); F_eg_tot_count += F_eg_count 
                
                for moment in range(L):     
                    for e in range(E):  
                        if iter%recomp_S[moment] == 0:
                            S_eg_new[moment,e], S_eg_count = recompute_S_eg(moment, e, phi_new, c_phi_new); S_eg_tot_count += S_eg_count

                if mode=='debug':
                    print tot_source(Q)
                    print tot_rxn(phi_new)
                    print leak_left(psi_left_new)
                    print leak_right(psi_right_new)
                    balance = tot_source(Q) - tot_rxn(phi_new) - leak_left(psi_left_new) - leak_right(psi_right_new)
                    print "balance = ", balance

                    B = residual(phi_new, c_phi_new, S_eg_new, F_eg=F_eg_new) 
                    print "residual = ", B
                        
                k_new   = k*np.linalg.norm(CF(c_phi_new,F_eg_new)[0])/np.linalg.norm(CF(c_phi,F_eg_new)[0]) 
                if k_exact == None:
		            k_error = np.abs(k_new - k)/k
                else:
                	k_error = abs(k_new - k_exact)/k_exact

                phi_new  /= np.linalg.norm(phi_new)
                phi_error = np.linalg.norm(phi_new - phi)/np.linalg.norm(phi)
                c_phi_new = coarse(phi_new)

                if iter > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                rho = 0
                print rho

                iter+=1

                figure_of_merit = H_tot_count + CS_tot_count + CF_tot_count + S_eg_tot_count + F_eg_tot_count
                tot_figure_of_merit += figure_of_merit
                LHS_cost += H_tot_count
                RHS_cost += CS_tot_count + CF_tot_count + S_eg_tot_count + F_eg_tot_count
                runtime = time.time()-t
                FOM_RT= tot_figure_of_merit/runtime
                print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" 
                                                                        %(iter,k,k_error,phi_error,t_Q,t_H,runtime) )
                print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  CS_count = %.2e  CF_count = %.2e  S_eg_tot_count = %.2e  F_eg_tot_count = %.2e\n\n"                                                     %(tot_figure_of_merit, RHS_cost, H_tot_count, CS_tot_count, CF_tot_count, S_eg_tot_count, F_eg_tot_count) )
                              
            if L_max > 1:
                iteration_dict.append({'runtime': runtime, 'k': np.copy(k_new), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error), 'residual': np.copy(B),
                                        'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                        'flux':np.copy(phi_new[:,0,:]), 'current':np.copy(phi_new[:,1,:])})
            else:
                iteration_dict.append({'runtime': runtime, 'k': np.copy(k_new), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error), 'residual': np.copy(B),
                                        'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                        'flux':np.copy(phi_new[:,0,:])})        
                
            t0 = time.time()
        runtime = time.time() - t
        return k_new, phi_new, psi_new/np.linalg.norm(psi_new), runtime, iteration_dict

    elif problem_type == "alpha":
        alpha = 0; d_alpha = 1.
        while (abs(d_alpha) > tol*(1-rho) or phi_error > tol*(1-rho)) and i < max_its:  
            phi_old = np.copy(phi)                
            phi     = phi_new / np.linalg.norm(phi_new)

            if discretization == "mg":
                Q = S(phi) + F(phi);   t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom, t_abs=alpha*invV())
                d_alpha = inner_prod(psi_new, S(phi_new)+F(phi_new)) / inner_prod(psi_new, invV(N)*psi_new) - \
                          inner_prod(psi_new, S(phi)+F(phi)) / inner_prod(psi_new, invV(N)*psi_new)
                alpha   += d_alpha 

            elif discretization == "cs":
                c_phi = np.copy(c_phi_new)                
                S_eg  = np.copy(S_eg_new)  
                F_eg  = np.copy(F_eg_new)  

                Q = CS(c_phi,S_eg) + CF(c_phi,F_eg);   t_Q = time.time() - t0
                phi_new, psi_new, psi_left_new, psi_right_new = invH(Q, geom, t_abs=alpha*invV())
                c_phi_new = coarse(phi_new);                     t_H = (time.time() - t0) - t_Q

                for e in range(E):  
                    update_S_eg = False  
                    update_F_eg = False  
                    for g in mapping[e]:
                        if i%1==0:
                            update_S_eg = True   
                        if i%1==0:
                            update_F_eg = True   
                    
                    if update_S_eg == True:
                        S_eg_new[e] = recompute_S_eg(e, phi_new, c_phi_new)
                    if update_F_eg == True:
                        F_eg_new[e] = recompute_F_eg(e, phi_new, c_phi_new)  

                d_alpha = inner_prod(psi_new, CS(c_phi_new,S_eg_new)+CF(c_phi_new,F_eg_new)) / inner_prod(psi_new, invV(N)*psi_new) - \
                          inner_prod(psi_new, CS(c_phi,S_eg)+CF(c_phi,F_eg)) / inner_prod(psi_new, invV(N)*psi_new)
                alpha   += d_alpha 

                alpha    = max(-max(mesh.sigt/mesh.invSpgrp), alpha)

                # for supercritical problems
                alpha    = max(0, alpha)

            phi_new  /= np.linalg.norm(phi_new)
            phi_error = np.linalg.norm(phi_new - phi)/np.linalg.norm(phi)

            i += 1
            if talk==True and i%1 == 0: print ("iteration %3i  alpha = %.10e  d_alpha = %.3e  phi_error = %.3e" %(i,alpha,d_alpha,phi_error) )

        runtime = time.time() - t
        return alpha*1e-8, phi_new, psi_new/np.linalg.norm(psi_new), runtime
###################################################################################################

