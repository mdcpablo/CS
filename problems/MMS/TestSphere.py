import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN_spherical as SN
import newGridXML2newGridObject
###############################################################################
mg_runtimes = []
cs_runtimes = []

'''
G = 1
CGs = [1]

mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('sphere_MMS_1.xml', 92001)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 1, num_cells=50, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 4, 1)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

bc = SN.BoundaryCondition(mesh, left='reflective', right='partial_current', J_right=1.)

phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'source', 'mg', mode='debug', L_max=1, tol=1e-5, max_its=100, DSA_opt=False)
print runtime_mg
mg_runtimes.append(runtime_mg)
'''
'''
k_exact = 1.
G = 100
CGs = [100]

mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001)}

heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8.7407, num_cells=200, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 32, 1)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

bc = SN.BoundaryCondition(mesh, left='vacuum', right='vacuum')

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'mg', mode='debug', L_max=3, tol=1e-5, max_its=40, k_exact=k_exact, DSA_opt=False)
print runtime_mg
mg_runtimes.append(runtime_mg)
'''
#'''
k_exact = 1.
G = 222
CGs = [222]

mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('FEDS_HMF001_'+str(G)+'.xml', 92001)}

#mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('hmf001_'+str(G)+'mg.xml', 92001)}


heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8.7407, num_cells=100, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 16, 1)
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

bc = SN.BoundaryCondition(mesh, left='vacuum', right='vacuum')

k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'mg', mode='debug', L_max=3, tol=1e-5, max_its=40, k_exact=k_exact, DSA_opt=False)
print runtime_mg
mg_runtimes.append(runtime_mg)
#'''





