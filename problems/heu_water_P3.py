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

dofs = [500]#[10, 25, 50, 100, 250, 500, 1000]

for G in dofs:
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

    '''
    if G == 10:
        CG = 2
        k_exact = 1.06148313
    elif G == 25:
        CG = 5
        k_exact = 0.736059811
    elif G == 50:
        CG = 10
        k_exact = 0.719275462
    elif G == 100:
        CG = 20
        k_exact = 0.713897010
    elif G == 250:
        CG = 50
        k_exact = 0.712639298
    elif G == 500:
        CG = 100
        k_exact = 0.712467677
    elif G == 1000:
        CG = 200
        k_exact = 0.712766535
    '''

    mesh = SN.GlobalMesh(mat_dict, [heu_1, water], 32, CG)
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    k, phi, psi, runtime_cs, iter_dict = SN.power_iterations(mesh, 'k', 'cs', mode='not debug', L_max=4, tol=2e-4, max_its=1000, k_exact=k_exact, recomp_F=2, recomp_S=[1,2,2,2])
    k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, 'k', 'mg', mode='not debug', L_max=4, tol=2e-4, max_its=1000, k_exact=k_exact)
    
    print runtime_cs
    print runtime_mg

    cs_runtimes.append(runtime_cs)
    mg_runtimes.append(runtime_mg)

plt.semilogx(dofs, [mg_runtimes[i]/cs_runtimes[i] for i in range(len(mg_runtimes))], 'bo')
plt.xlabel('Number of Energy Groups')
plt.ylabel('Simulation Runtime')
plt.xlim([5, 2000])
plt.ylim([0.5, 6])
plt.show()

plt.loglog(dofs, mg_runtimes, 'bo')
plt.loglog(dofs, cs_runtimes, 'ro')
plt.xlabel('Number of Energy Groups')
plt.ylabel('Simulation Runtime')
plt.xlim([5, 2000])
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



