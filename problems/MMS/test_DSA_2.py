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

G = 10

mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001), 
        'water': newGridXML2newGridObject.dict_to_object('water_'+str(G)+'mg.xml', 11111)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 4, num_cells=10, log_option=False)
water = SN.ZoneSpatialMesh('water', 4, 8, num_cells=10, log_option=False)

if G == 10:
    CG = 5
    k_exact = 1.16562269
elif G == 25:
    CG = 5
    k_exact = 0.906056786
elif G == 50:
    CG = 5
    k_exact = 0.8928886
elif G == 100:
    CG = 5
    k_exact = 0.885535499
elif G == 250:
    CG = 5
    k_exact = 0.8754518
elif G == 500:
    CG = 5
    k_exact = 0.867832
elif G == 1000:
    CG = 5
    k_exact = 0.8594517

mesh = SN.GlobalMesh(mat_dict, [heu_1, water], 32, CG)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, 'k', 'mg', mode='not debug', L_max=4, tol=2e-4, max_its=1000, k_exact=k_exact)

print runtime_mg

mg_runtimes.append(runtime_mg)

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


