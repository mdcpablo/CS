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

G = 1000
if G == 10:
    CG = 5
    k_exact = 1.71529932
elif G == 25:
    CG = 5
    k_exact = 1.21767073
elif G == 50:
    CG = 5
    k_exact = 1.19286621
elif G == 100:
    CG = 5
    k_exact = 1.18548877
elif G == 250:
    CG = 5
    k_exact = 1.18369332
elif G == 500:
    CG = 5
    k_exact = 1.18344208
elif G == 1000:
    CG = 5
    k_exact = 1.18337553

mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8, num_cells=20, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 32, CG)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, 'k', 'mg', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=k_exact)
print runtime_mg
mg_runtimes.append(runtime_mg)

R = [1,2,3,4,5]
for r in R:
    k, phi, psi, runtime_cs, iter_dict = SN.power_iterations(mesh, 'k', 'cs', mode='not debug', L_max=1, tol=1e-5, max_its=1000, k_exact=k_exact, recomp_F=12, recomp_S=[r])
    
    print runtime_cs
   
    cs_runtimes.append(runtime_cs)

#plt.semilogx(R, [mg_runtimes[0]/cs_runtimes[i] for i in range(len(cs_runtimes))], 'bo')
#plt.xlabel('R')
#plt.ylabel('Simulation Runtime')
#plt.xlim([0,10])
#plt.ylim([0, 2])
#plt.show()

plt.plot(R, cs_runtimes, 'ro')
plt.xlabel('R')
plt.ylabel('Simulation Runtime')
plt.xlim([0,10])
plt.ylim([0, 10000])
plt.show()

#plt.loglog([1], mg_runtimes, 'bo')
plt.loglog(R, cs_runtimes, 'ro')
plt.xlabel('R')
plt.ylabel('Simulation Runtime')
plt.xlim([0,10])
plt.ylim([0.005, 10000])
plt.show()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('CS residual')
plt.show()

plt.semilogy(np.arange(len(iter_dict)), [dic['residual'] for dic in iter_dict])
plt.xlabel('iteration number')
plt.ylabel('$k$ error')
plt.show()



