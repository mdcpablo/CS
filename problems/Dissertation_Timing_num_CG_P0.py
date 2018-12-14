import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN as SN
import newGridXML2newGridObject
###############################################################################
mg_runtimes = []
cs_runtimes = []

k_exact = 1.18337553
G = 1000
CGs = [1,5,10,25,50,100,250]

'''
mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8, num_cells=20, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 32, 1)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, 'k', 'mg', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=k_exact)
print runtime_mg
mg_runtimes.append(runtime_mg)
'''

for CG in CGs:
    mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001)}

    heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8, num_cells=20, log_option=False)

    mesh = SN.GlobalMesh(mat_dict, [heu_1], 32, CG)
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    k, phi, psi, runtime_cs, iter_dict = SN.power_iterations(mesh, 'k', 'cs', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=k_exact, recomp_F=18, recomp_S=[3])
    
    print runtime_cs
   
    cs_runtimes.append(runtime_cs)

#plt.semilogx(R, [mg_runtimes[0]/cs_runtimes[i] for i in range(len(cs_runtimes))], 'bo')
#plt.xlabel('R')
#plt.ylabel('Simulation Runtime')
#plt.xlim([0,10])
#plt.ylim([0, 2])
#plt.show()
#plt.close()

plt.plot(CGs, cs_runtimes, 'ko')
plt.xlabel('Number of Coarse Groups')
plt.ylabel('Simulation Runtime')
plt.xlim([0,300])
plt.ylim([0, 10000])
plt.show()
plt.close()

plt.loglog(CGs, cs_runtimes, 'ko')
plt.xlabel('Number of Coarse Groups')
plt.ylabel('Simulation Runtime')
plt.xlim([0.3,300])
plt.ylim([0, 10000])
plt.show()
plt.close()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('CS residual')
plt.show()
plt.close()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('$k$ error')
plt.show()
plt.close()


