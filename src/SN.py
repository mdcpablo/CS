import numpy as np
import time
from scipy import interpolate, special
from numpy.polynomial.legendre import leggauss
###############################################################################
# obtain fission spectrum for U-235, E must be in (MeV) 
default_chi = lambda E: 0.4865*np.sinh(np.sqrt(2*E))*np.exp(-E)
###############################################################################
# obtain speed in (cm/s) for a particular energy (MeV) 
vel = lambda E: np.sqrt(2.*E/938.280)*3e10 
###############################################################################
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
###############################################################################
class GlobalMesh:
    def __init__(self, mat_dict, zones, num_angles, num_c_grps):
        self = self
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
        # precomputing legendre polynomial value for order ell for direction mu_m
        for moment in range(self.nlgndr):
            for m in range(self.num_angles):
                self.legendre[moment,m] = special.legendre(moment)(self.mu[m]) 

        self.mat_sigt = {}
        self.mat_sigf = {}
        self.mat_p = {}
        self.mat_invSpgrp = {}

        for mat in sorted(mat_dict.keys()):
            self.mat_sigt[mat] = mat_dict[mat].sigt
            self.mat_sigf[mat] = mat_dict[mat].sigf
            self.mat_p[mat] = mat_dict[mat].p  
            self.mat_invSpgrp[mat] = mat_dict[mat].invSpgrp

        self.sigt        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.sigt[:,i]          = self.mat_sigt[self.cell_mat[i]]

        self.invSpgrp        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.invSpgrp[:,i]          = self.mat_invSpgrp[self.cell_mat[i]]

        self.num_c_grps = num_c_grps
        self.num_grps = len(self.emid)
        sub_g = int(np.ceil( self.num_grps / self.num_c_grps ))

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

        print self.mapping

        '''
        self.mat_sigt = {}
        self.mat_sigf = {}
        self.mat_p = {}
        if hasattr(mat_dict[sorted(mat_dict.keys())[0]], 'coarse_sigf'):
            self.mat_coarse_sigf = {}
        if hasattr(mat_dict[sorted(mat_dict.keys())[0]], 'coarse_p'):
            self.mat_coarse_p = {}
            self.mapping = mat_dict[sorted(mat_dict.keys())[0]].mapping
        self.mat_invSpgrp = {}

        for mat in sorted(mat_dict.keys()):
            self.mat_sigt[mat] = mat_dict[mat].sigt
            self.mat_sigf[mat] = mat_dict[mat].sigf
            self.mat_p[mat] = mat_dict[mat].p  
            if hasattr(mat_dict[sorted(mat_dict.keys())[0]], 'coarse_sigf'):
                self.mat_coarse_sigf[mat] = mat_dict[mat].coarse_sigf
            if hasattr(mat_dict[sorted(mat_dict.keys())[0]], 'coarse_p'):
                self.mat_coarse_p[mat] = mat_dict[mat].coarse_p  
            self.mat_invSpgrp[mat] = mat_dict[mat].invSpgrp

        self.sigt        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.sigt[:,i]          = self.mat_sigt[self.cell_mat[i]]

        self.invSpgrp        = np.zeros((self.num_grps, self.num_cells))
        for i in range(self.num_cells):
            self.invSpgrp[:,i]          = self.mat_invSpgrp[self.cell_mat[i]]
        '''
    
        
        '''self.sigt        = np.zeros((self.num_grps, self.num_cells))
        self.sigf        = np.zeros((self.num_grps, self.num_grps, self.num_cells))
        self.p           = np.zeros((self.nlgndr, self.num_grps, self.num_grps, self.num_cells))
        self.coarse_sigf = np.zeros((len(self.mapping), len(self.mapping), self.num_cells))
        self.coarse_p    = np.zeros((self.nlgndr, len(self.mapping), len(self.mapping), self.num_cells))
        for i in range(self.num_cells):
            self.sigt[:,i]          = self.mat_sigt[self.cell_mat[i]]
            self.sigf[:,:,i]        = self.mat_sigf[self.cell_mat[i]]
            self.coarse_sigf[:,:,i] = self.mat_coarse_sigf[self.cell_mat[i]]
            for moment in range(self.nlgndr):
                self.p[moment,:,:,i] = self.mat_p[self.cell_mat[i]][moment][:,:]
                self.coarse_p[moment,:,:,i] = self.mat_coarse_p[self.cell_mat[i]][moment][:,:]
        '''


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
###############################################################################
class BoundaryCondition:
    def __init__(self, mesh, left='vacuum', right='vacuum', phi_left=0, phi_right=0, psi_left_mu1=0, psi_right_mu1=0, time_profile=0):
        self = self
        self.left = left
        self.right = right  

        G, N = mesh.num_grps, mesh.num_angles
        self.psi_left = np.zeros((G, N))  
        self.psi_right = np.zeros((G, N))  

        if left == 'isotropic':
            self.psi_left[:,N/2:N] = phi_left / sum( mesh.w )
        if right == 'isotropic':
            self.psi_right[:,0:mesh.N/2] = phi_boundary / sum( mesh.w )

        if left == 'mu1':
            self.psi_left[:,N-1] = (1. - mu[N-1])/(mu[N-1] - mu[N-2]) * psi_left_mu1 / w[N-1]
        if right == 'mu1':
            self.psi_right[:,0] = (1. + mu[0])/(-mu[0] + mu[1]) * psi_right_mu1 / w[0] 
###############################################################################     
def power_iterations(mesh, eigenvalue, discretization, mode='normal', L_max=9, S_tol=1e-6, F_tol=1e-6, max_its=1e3, tol=1e-6, talk=True):
    G, N, I = mesh.num_grps, mesh.num_angles, mesh.num_cells
    L = min(mesh.nlgndr, L_max)

    def dot(a,b):
        ab = 0
        for i in range(len(a)):
            ab += a[i]*b[i]
        return ab   

    def take_moments(psi):
        phi = np.zeros((G, L, I))
        for g in range(G):
            for moment in range(L):
                for m in range(N):
                    phi[g,moment,:] += psi[g,m,:]*mesh.legendre[moment,m]*mesh.w[m]
        return phi   

    def inner_prod(x,Ax):
        inner_prod = 0
        for g in range(G):
            for m in range(N):
                for i in range(I):
                    inner_prod += x[g,m,i]*Ax[g,m,i]
        return inner_prod  

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
        

    def invH(Q, t_abs=0):
        # sweeps across all cells for all energies and directions
        psi = np.zeros((G,N,I)); psi_left = np.zeros((G,N,I)); psi_right = np.zeros((G,N,I)); count=0
        sigt = mesh.sigt + t_abs # adding time absorption
        for g in range(G):
            for m in range(N):
                psi_L = 0.
                psi_R = 0.
                psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:] = sweep_step(I, mesh.dx, mesh.mu[m], sigt[g,:], Q[g,m,:], psi_L, psi_R)
                #psi[g,m,:], psi_left[g,m,:], psi_right[g,m,:] = sweep_diamond(I, mesh.dx, mesh.mu[m], mesh.sigt[g,:], Q[g,m,:], psi_L, psi_R)
                count += I #15*I
        if mode == 'debug':
            return take_moments(psi), psi, count
        else:
            return take_moments(psi), psi

    def invH_inf(Q, t_abs=0):
        # infinite-medium Boltzmann operator (no streaming)
        sigt = mesh.sigt + t_abs # adding time absorption
        psi = np.zeros((G,N,I)); count=0
        for g in range(G):
            for m in range(N):
                psi[g,m,:] += Q[g,m,:]/sigt[g,:]
                count += 1
        if mode == 'debug':
            return take_moments(psi), psi, count
        else:
            return take_moments(psi), psi  
  

    def sweep_step(I, dx, mu, sigt, Q, psi_L, psi_R):
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

    def sweep_diamond(I, dx, mu, sigt, Q, psi_L, psi_R):
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

    def S(phi):
        S = np.zeros((G,N,I)); count=0
        for i in range(I):
            p = mesh.mat_p[mesh.cell_mat[i]]
            for moment in range(L):
                sum_over_moment = np.zeros(G)
                for g in range(G):
                    for gp in range(G):
                        sum_over_moment[g] += p[moment][gp,g]*phi[gp,moment,i]
                        count += 1
                    for m in range(N):
                        S[g,m,i] += (2.*moment+1.)/2.*mesh.legendre[moment,m]*sum_over_moment[g]
                        count += 1
        if mode == 'debug':
            return S, count
        else:
            return S

    def F(phi):
        F = np.zeros((G,N,I)); count = 0
        for i in range(I):
            sigf = mesh.mat_sigf[mesh.cell_mat[i]]
            sum_fiss = np.zeros(G)
            for g in range(G):
                for gp in range(G):
                    sum_fiss[g] += 0.5*sigf[gp,g]*phi[gp,0,i] 
                    count += 1
                for m in range(N):
                    F[g,m,i] += sum_fiss[g]
                    count += 1
        if mode == 'debug':
            return F, count
        else:
            return F

    if discretization == 'cs':
        mapping = mesh.mapping
        E = len(mapping)

        def coarse(phi):
            coarse_phi = np.zeros((E,L,I))
            for e in range(E):  
                for g in mapping[e]:  
                    for moment in range(L): 
                        for i in range(I):
                            coarse_phi[e,moment,i] += phi[g,moment,i]
            return coarse_phi

        def recompute_S_eg(moment, e, phi, coarse_phi):
            S_moment_e = np.zeros((G,I)); count=0
            for i in range(I):
                p = mesh.mat_p[mesh.cell_mat[i]]
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for ep in range(E):
                        denom += coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]+1e-16  #denom += (coarse_p[moment][ep,e]+1e-14)*(coarse_phi[ep,moment,i]+1e-14) 
                        count += 1
                        for gp in mapping[ep]:   
                            numer += p[moment][gp,g]*phi[gp,moment,i]+1e-16  #numer += (p[moment][gp,g]+1e-14)*(phi[gp,moment,i]+1e-14) 
                            count += 1
                        S_moment_e[g,i] = numer/denom
            #if talk==True: print "S_eg was recomputed"
            if mode == 'debug':
                return S_moment_e, count
            else:
                return S_moment_e

        def recompute_F_eg(e, phi, coarse_phi):
            F_e = np.zeros((G,I)); count=0
            for i in range(I):
                sigf = mesh.mat_sigf[mesh.cell_mat[i]]
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                for g in mapping[e]:   
                    numer = 0. 
                    denom = 0.
                    for ep in range(E):
                        denom += coarse_sigf[ep,e]*coarse_phi[ep,0,i]+1e-16  #denom += (coarse_sigf[ep,e]+1e-14)*(coarse_phi[ep,0,i]+1e-14) 
                        count += 1
                        for gp in mapping[ep]:   
                            numer += sigf[gp,g]*phi[gp,0,i]+1e-16  #numer += (sigf[gp,g]+1e-14)*(phi[gp,0,i]+1e-14) 
                            count += 1
                        F_e[g,i] = numer/denom
            #if talk==True: print "F_eg was recomputed"
            if mode == 'debug':
                return F_e, count
            else:
                return F_e

        '''
        def S(phi):
            S = np.zeros((G,N,I)); count=0
            for i in range(I):
                p = mesh.mat_p[mesh.cell_mat[i]]
                for moment in range(L):
                    sum_over_moment = np.zeros(G)
                    for g in range(G):
                        for gp in range(G):
                            sum_over_moment[g] += p[moment][gp,g]*phi[gp,moment,i]
                            count += 1
                        for m in range(N):
                            S[g,m,i] += (2.*moment+1.)/2.*mesh.legendre[moment,m]*sum_over_moment[g]
                            count += 1
            if mode == 'debug':
                return S, count
            else:
                return S
        '''

        def CS(coarse_phi, S_eg):
            CS = np.zeros((G,N,I)); count=0
            for i in range(I):
                coarse_p = mesh.mat_coarse_p[mesh.cell_mat[i]]
                for moment in range(L):
                    sum_over_moment = np.zeros(G)
                    for e in range(E):
                        for g in mapping[e]: 
                            for ep in range(E):
                                sum_over_moment[g] += S_eg[moment,e,g,i]*coarse_p[moment][ep,e]*coarse_phi[ep,moment,i]
                                count += 1
                            for m in range(N):
                                CS[g,m,i] += (2.*moment+1.)/2.*mesh.legendre[moment,m]*sum_over_moment[g]
                                count += 1
            if mode == 'debug':
                return CS, count
            else:
                return CS

        def CF(coarse_phi, F_eg):
            CF = np.zeros((G,N,I)); count=0
            for i in range(I):
                coarse_sigf = mesh.mat_coarse_sigf[mesh.cell_mat[i]]
                sum_fiss = np.zeros(G)
                for e in range(E):
                    for g in mapping[e]: 
                        for ep in range(E):
                            sum_fiss[g] += 0.5*F_eg[e,g,i]*coarse_sigf[ep,e]*coarse_phi[ep,0,i]
                            count += 1
                        for m in range(N):
                            CF[g,m,i] += sum_fiss[g]
                            count += 1
            if mode == 'debug':
                return CF, count
            else:
                return CF

    t = time.time()
    phi = np.ones((G,L,I))
    phi_new = np.ones((G,L,I))
    phi_error = 1.;  i = 0; t0 = time.time(); iteration_dict = []    
    rho = 0; # spectral radius
    if discretization == 'cs':
        c_phi_new = coarse(phi_new)
        mapping = mesh.mapping
        S_eg_new = np.zeros((L,E,G,I))
        F_eg_new = np.zeros((E,G,I))
        for e in range(E):
            if mode == 'debug':
                for moment in range(L):
                    S_eg_new[moment, e], S_eg_count = recompute_S_eg(moment, e, phi_new, c_phi_new);      S_eg_tot_count = S_eg_count
                F_eg_new[e], F_eg_count = recompute_F_eg(e, phi_new, c_phi_new);      F_eg_tot_count = F_eg_count 
            else:      
                for moment in range(L):
                    S_eg_new[moment, e] = recompute_S_eg(moment, e, phi_new, c_phi_new)
                F_eg_new[e] = recompute_F_eg(e, phi_new, c_phi_new)
        phi_ratio_at_prev_S_eg=np.zeros(G)
        phi_ratio_at_prev_F_eg=np.zeros(G)
 
    if eigenvalue == "k":
        print "\nCommencing Power iterations to calculate k:"
        k_new = 1.; k_error = 1.; tot_figure_of_merit=0; LHS_cost = 0; RHS_cost = 0;
        while (k_error > tol*(1-rho) or phi_error > tol*(1-rho)) and i < max_its+1 or (i-1)%4 != 0: 
            phi_old = np.copy(phi)                
            phi     = np.copy(phi_new)
            k       = np.copy(k_new)

            if discretization == "mg":
                S_tot_count = 0; F_tot_count = 0; H_tot_count = 0

                if mode == 'debug':
                    S_phi, S_count = S(phi);                        S_tot_count += S_count; 
                    F_phi, F_count = F(phi);                        F_tot_count += F_count;
                    Q = S_phi + F_phi/k;                            t_Q = time.time() - t0
                    phi_new, psi_new, H_count = invH(Q);            H_tot_count += H_count;  t_H = (time.time() - t0) - t_Q
                else:
                    Q = S(phi) + F(phi)/k;                          t_Q = time.time() - t0
                    phi_new, psi_new = invH(Q);                     t_H = (time.time() - t0) - t_Q
               
                k_new   = k*np.linalg.norm(F(phi_new)[0])/np.linalg.norm(F(phi)[0]) 
                #k_new = k*inner_prod(phi_new, F(phi_new))/ inner_prod(phi_new, (H() - S(phi_new)))

                phi_new   /= np.linalg.norm(phi_new)
                phi_error = np.linalg.norm(phi_new - phi)

                k_error   = abs(k_new - 1.1542032) #np.abs(k_new - k)/k 

                if i > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                print rho

                i += 1

                if mode=='debug':
                    figure_of_merit = H_tot_count + S_tot_count + F_tot_count
                    tot_figure_of_merit += figure_of_merit
                    LHS_cost += H_tot_count
                    RHS_cost += S_tot_count + F_tot_count 
                    runtime = time.time()-t
                    FOM_RT= tot_figure_of_merit/runtime
                    print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" 
                                                                            %(i,k,k_error,phi_error,t_Q,t_H,runtime) )
                    print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  S_count = %.2e  F_count = %.2e\n\n" %(tot_figure_of_merit,RHS_cost, H_tot_count, S_tot_count, F_tot_count))
                                                                                        
                elif talk==True and i%1 == 0: 
                    runtime = time.time()-t
                    print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f" 
                                                                                        %(i,k,k_error,phi_error,t_Q,t_H,runtime))
            elif discretization == "cs":
                CS_tot_count = 0; CF_tot_count = 0; H_tot_count = 0; S_eg_tot_count = 0; F_eg_tot_count = 0

                c_phi = np.copy(c_phi_new)                
                S_eg  = np.copy(S_eg_new)  
                F_eg  = np.copy(F_eg_new)               

                if mode == 'debug':
                    CS_phi, CS_count = CS(c_phi,S_eg);                CS_tot_count += CS_count;
                    CF_phi, CF_count = CF(c_phi,F_eg);                CF_tot_count += CF_count;
                    Q = CS_phi + CF_phi/k;                            t_Q = time.time() - t0
                    phi_new, psi_new, H_count = invH(Q);              H_tot_count += H_count; 
                    c_phi_new = coarse(phi_new);                      t_H = (time.time() - t0) - t_Q
                else:
                    Q = CS(c_phi,S_eg) + CF(c_phi,F_eg)/k;            t_Q = time.time() - t0
                    phi_new, psi_new = invH(Q) 
                    c_phi_new = coarse(phi_new);                      t_H = (time.time() - t0) - t_Q
 
                for e in range(E):  
                    update_S_eg = False  
                    update_F_eg = False  
                    for g in mapping[e]:
                        if i%2==0:
                            update_S_eg = True   
                        if i%4==0:
                            update_F_eg = True   
                    
                    if mode == 'debug':
                        for moment in range(L):
                            if update_S_eg == True:
                                S_eg_new[moment,e], S_eg_count = recompute_S_eg(moment, e, phi_new, c_phi_new);      S_eg_tot_count += S_eg_count
                        if update_F_eg == True:
                            F_eg_new[e], F_eg_count = recompute_F_eg(e, phi_new, c_phi_new);      F_eg_tot_count += F_eg_count 
                    else:      
                        for moment in range(L):
                            if update_S_eg == True:
                                S_eg_new[moment,e] = recompute_S_eg(moment, e, phi_new, c_phi_new)
                        if update_F_eg == True:
                            F_eg_new[e] = recompute_F_eg(e, phi_new, c_phi_new)

                k_new   = k*np.linalg.norm(CF(c_phi_new,F_eg_new)[0])/np.linalg.norm(CF(c_phi,F_eg_new)[0]) 
                k_error = abs(k_new - 1.1542032) #np.abs(k_new - k)/k 

                phi_new  /= np.linalg.norm(phi_new)
                phi_error = np.linalg.norm(phi_new - phi)/np.linalg.norm(phi)

                if i > 1:
                    rho = np.linalg.norm( phi_new - phi ) / np.linalg.norm( phi - phi_old )
                print rho

                i+=1

                if mode=='debug':
                    figure_of_merit = H_tot_count + CS_tot_count + CF_tot_count + S_eg_tot_count + F_eg_tot_count
                    tot_figure_of_merit += figure_of_merit
                    LHS_cost += H_tot_count
                    RHS_cost += CS_tot_count + CF_tot_count + S_eg_tot_count + F_eg_tot_count
                    runtime = time.time()-t
                    FOM_RT= tot_figure_of_merit/runtime
                    print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f\n" 
                                                                            %(i,k,k_error,phi_error,t_Q,t_H,runtime) )
                    print ("          tot_figure_of_merit = %.2e  RHS_cost = %.2e  H_count = %.2e  CS_count = %.2e  CF_count = %.2e  S_eg_tot_count = %.2e  F_eg_tot_count = %.2e\n\n"                                                     %(tot_figure_of_merit, RHS_cost, H_tot_count, CS_tot_count, CF_tot_count, S_eg_tot_count, F_eg_tot_count) )
                                                                                        
                elif talk==True and i%1 == 0: 
                    runtime = time.time()-t
                    print ("iter %3i  k = %.8e  k_error = %.1e  phi_error = %.1e  t_Q:t_H = %.1f:%.1f  tot_runtime = %.1f" 
                                                                                        %(i,k,k_error,phi_error,t_Q,t_H,runtime))
            '''
            elif discretization == "xcs":
                Q = XCS(phi) + XCF(phi)/k
                phi_new, psi_new = invH(Q)
                               
                k_new   = k*np.linalg.norm(F(phi_new)[0])/np.linalg.norm(F(phi)[0]) 
                #k_new = k*inner_prod(phi_new, F(phi_new))/ inner_prod(phi_new, (H() - S(phi_new)))

                phi_new   /= np.linalg.norm(phi_new)
                phi_error = np.linalg.norm(phi_new - phi)

                k_error   = np.abs(k_new - k) 

                i += 1
            '''

            if mode == 'debug':
                if L_max > 1:
                    iteration_dict.append({'runtime': runtime, 'k': np.copy(k_new), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error),
                                            'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                            'flux':np.copy(phi_new[:,0,:]), 'current':np.copy(phi_new[:,1,:])})
                else:
                    iteration_dict.append({'runtime': runtime, 'k': np.copy(k_new), 'k_error': np.copy(k_error), 'phi_error': np.copy(phi_error),
                                            'RHS_cost': RHS_cost, 'total_cost': tot_figure_of_merit, 
                                            'flux':np.copy(phi_new[:,0,:])})        
            else:
                iteration_dict = [{}]
                
            t0 = time.time()
        runtime = time.time() - t
        return k_new, phi_new, psi_new/np.linalg.norm(psi_new), runtime, iteration_dict

    elif eigenvalue == "alpha":
        alpha = 0; d_alpha = 1.
        while (abs(d_alpha) > tol*(1-rho) or phi_error > tol*(1-rho)) and i < max_its:  
            phi_old = np.copy(phi)                
            phi     = phi_new / np.linalg.norm(phi_new)

            if discretization == "mg":
                Q = S(phi) + F(phi);   t_Q = time.time() - t0
                phi_new, psi_new = invH(Q, t_abs=alpha*invV())
                d_alpha = inner_prod(psi_new, S(phi_new)+F(phi_new)) / inner_prod(psi_new, invV(N)*psi_new) - \
                          inner_prod(psi_new, S(phi)+F(phi)) / inner_prod(psi_new, invV(N)*psi_new)
                alpha   += d_alpha 

            elif discretization == "cs":
                c_phi = np.copy(c_phi_new)                
                S_eg  = np.copy(S_eg_new)  
                F_eg  = np.copy(F_eg_new)  

                Q = CS(c_phi,S_eg) + CF(c_phi,F_eg);   t_Q = time.time() - t0
                phi_new, psi_new = invH(Q, t_abs=alpha*invV())
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
###############################################################################

