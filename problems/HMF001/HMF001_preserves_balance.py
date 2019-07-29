import os
import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN
import newGridXML2newGridObject
###############################################################################
def output_run_details(xs_file, mesh, L_max, comment, k, runtime_mg, iter_dict):
    outdir = xs_file.split('.')[0]    
    if not os.path.exists('outputs/'+outdir): 
        os.makedirs('outputs/'+outdir)  
    P = min(mesh.nlgndr, L_max)
    output_file = ('outputs/%s/%s_I%i_N%i_P%i.out' %(outdir,outdir,mesh.num_cells,mesh.num_angles,P))
    with open(output_file,'w') as f1:
        f1.write('xs_file = %s \n' %xs_file)
        f1.write('G = %s \n' %mesh.num_grps)
        f1.write('I = %s \n' %mesh.num_cells)
        f1.write('N = %s \n' %mesh.num_angles)
        f1.write('P = %s \n' %P)
        f1.write('\n# COMMENTS\n')
        f1.write('%s\n' %comment)
        f1.write('\n# TIMING\n')
        f1.write('runtime = %.2f \n' %runtime_mg)
        f1.write('\n# RESULTS\n')
        f1.write('k = %.14f \n' %k)

        flux = iter_dict[-1]['flux']

        f1.write('phi_array = np.array([\n[')
        for g in range(mesh.num_grps):
            for i in range(mesh.num_cells):
                if i < mesh.num_cells-1:
                    f1.write('%.14e, ' %flux[g,i])
                else:
                    f1.write('%.14e' %flux[g,i])
            if g < mesh.num_grps-1:
                f1.write('],\n')
            else:
                f1.write(']])\n')

        f1.write('\n# FLUX\n')
        for g in range(mesh.num_grps):
            for i in range(mesh.num_cells):
                f1.write('%.14e ' %flux[g,i])
            if g < mesh.num_grps-1:
                f1.write('\n')
# -----------------------------------------------------------------------------
def run(xs_file, I=2, N=2, L_max=1, note=''):
    mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object('xs/'+xs_file, 92001)}

    #heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8.7407, num_cells=I, log_option=False)
    heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8.7407, num_cells=I, log_option=False)

    mesh = SN.GlobalMesh(mat_dict, [heu_1], N, 1, 'sphere')
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    bc = SN.BoundaryCondition(mesh, right='vacuum')
     
    recomp_F = 1
    recomp_S = [1, 16, 16, 16, 16, 16, 16, 16, 16]

    k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'cs', mode='normal', L_max=L_max, tol=1e-5, max_its=1000, k_exact=k_exact, DSA_opt=False, recomp_F=recomp_F, recomp_S=recomp_S)

    comment = 'cs, recomp_F='+str(recomp_F)+', recomp_S='+str(recomp_S)
    #output_run_details(xs_file, mesh, L_max, comment, k, runtime_mg, iter_dict)

    return k, phi, psi, runtime_mg, iter_dict
# -----------------------------------------------------------------------------
def run_dofs(xs_filename_left, dofs, xs_filename_right, I=2, N=2, L_max=1, note=''):
    k_list = []
    k_error_list = []
    for g in dofs:        
        k, phi, psi, runtime_mg, iter_dict = run(xs_filename_left+str(g)+xs_filename_right, I=I, N=N, L_max=L_max, note=note)
        k_list.append(np.copy(k))
        k_error_list.append(np.abs(k-k_exact))
    return k_list, k_error_list
# -----------------------------------------------------------------------------
k_exact = 1.10315620 #400, cs26=14.3, cs2=42., cs3=28.7, mg=42.3
#dofs = [400,600] #[100,200,400,600,800]
ps = [3,5]
sn = [32,64,128,256]
Is = [100,250,500,1000]

# needed for accuracy:
# 250 cells, P5, S128 
# -----------------------------------------------------------------------------
dofs = [100]
#dofs = [100]
mg_k, mg_k_error = run_dofs('HMF001_',dofs,'mg.xml',I=8,N=8,L_max=3)#, I=5, N=4, L_max=1, note='MG')

#22.9
#28.9

#25.3
#34.6




