import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN as SN
import newGridXML2newGridObject
###############################################################################
mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_800mg.xml', 92001)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8, num_cells=1, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 2, 50)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, 'k', 'mg', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=0.712766535)
k, phi, psi, runtime_cs, iter_dict = SN.power_iterations(mesh, 'k', 'cs', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=0.712766535, recomp_F=8, recomp_S=[2])
print runtime_mg
print runtime_cs

forward_flux = np.zeros((mesh.num_grps,mesh.num_cells))
backward_flux = np.zeros((mesh.num_grps,mesh.num_cells))
for g in range(mesh.num_grps):
    for m in range(mesh.num_angles):
        for i in range(mesh.num_cells):
            if mesh.mu[m] > 0:
                forward_flux[g,i] += psi[g,m,i]*mesh.w[m]
            if mesh.mu[m] < 0:
                backward_flux[g,i] += psi[g,m,i]*mesh.w[m]

plt.loglog(mesh.emid, forward_flux[:,0], label='first-cell forward-spectrum')
plt.loglog(mesh.emid, backward_flux[:,0], label='first-cell backward-spectrum')
plt.loglog(mesh.emid, forward_flux[:,-1], label='last-cell forward-spectrum')
plt.loglog(mesh.emid, backward_flux[:,-1], label='last-cell backward-spectrum')
#plt.show()
plt.close()

forward_flux_3g = np.zeros((3,mesh.num_cells))
backward_flux_3g = np.zeros((3,mesh.num_cells))
for g in range(mesh.num_grps):
    for m in range(mesh.num_angles):
        for i in range(mesh.num_cells):
            if mesh.mu[m] > 0:
                if mesh.emid[g] <= 1e-6:
                    forward_flux_3g[2,i] += forward_flux[g,i]
                elif 1e-6 < mesh.emid[g] < 0.1:
                    forward_flux_3g[1,i] += forward_flux[g,i]
                elif mesh.emid[g] >= 0.1:
                    forward_flux_3g[0,i] += forward_flux[g,i]
            if mesh.mu[m] < 0:
                if mesh.emid[g] <= 1e-6:
                    backward_flux_3g[2,i] += backward_flux[g,i]
                elif 1e-6 < mesh.emid[g] < 0.1:
                    backward_flux_3g[1,i] += backward_flux[g,i]
                elif mesh.emid[g] >= 0.1:
                    backward_flux_3g[0,i] += backward_flux[g,i]


plt.plot(mesh.x_mids, forward_flux_3g[0,:], label='fast_flux_plus')
plt.plot(mesh.x_mids, forward_flux_3g[1,:], label='epithermal_flux_plus')
plt.plot(mesh.x_mids, forward_flux_3g[2,:], label='thermal_flux_plus')
plt.plot(mesh.x_mids, backward_flux_3g[0,:], label='fast_flux_minus')
plt.plot(mesh.x_mids, backward_flux_3g[1,:], label='epithermal_flux_minus')
plt.plot(mesh.x_mids, backward_flux_3g[2,:], label='thermal_flux_minus')
plt.yscale('log')
plt.legend()
#plt.show()
plt.close()

flux_3g = np.zeros((3,mesh.num_cells))
flux_3g[0] = forward_flux_3g[0] + backward_flux_3g[0]
flux_3g[1] = forward_flux_3g[1] + backward_flux_3g[1]
flux_3g[2] = forward_flux_3g[2] + backward_flux_3g[2]
tot_flux = sum(flux_3g)

plt.plot(mesh.x_mids, tot_flux, 'k')
plt.show()
plt.close()

spectrum = 0
for i in range(mesh.num_cells):
    spectrum += phi[:,0,i].flatten()

plt.loglog(mesh.emid, spectrum, 'k')
plt.show()
plt.close()

#plt.plot(mesh.x_mids, phi[0,0,:])
#plt.show()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('CS residual')
plt.show()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('$k$ error')
plt.show()



