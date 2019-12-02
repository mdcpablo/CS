import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN 
import newGridXML2newGridObject
###############################################################################
#mat_dict = {'heu20': newGridXML2newGridObject.dict_to_object('xs/CS_300_5.xml', 92001)}
#mat_dict = {'heu20': newGridXML2newGridObject.dict_to_object('xs/BarnfireXS_HEU20_30.xml', 92001)}
#mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('newGrid_hmf001_cs_25.xml', 92235)}
mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('MMS_2b.xml', 92235)}


heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 1, num_cells=1000, log_option=False)
#heu20_1 = SN.ZoneSpatialMesh('heu20', 0, 25, num_cells=5, log_option=False)

mesh = SN.GlobalMesh(mat_dict, [heu_1], 8, 1, 'slab')
mesh.print_energies()
mesh.print_angles()
mesh.print_space(v=0)

#bc = 0
#Q_ext = np.zeros((mesh.num_grps,mesh.num_angles,mesh.num_cells))
#phi = np.zeros((mesh.num_grps,mesh.nlgndr,mesh.num_cells))
#phi = SN.one_source_iteration(mesh, bc, phi, Q_ext)
#plt.plot(mesh.x_mids, phi[15,0,:])
#plt.show()

phi, psi, runtime, iter_dict = SN.power_iterations(mesh, 'source', 'cs', mode='debug', L_max=1, tol=1e-15, max_its=1000)

plt.plot(mesh.x_mids, phi[0,0,:])
plt.show()


error = phi[0,0,:] - 2.*np.sin(np.pi*mesh.x_mids/(mesh.x_edges[-1]-mesh.x_edges[0]))
print error
print np.linalg.norm(error)


