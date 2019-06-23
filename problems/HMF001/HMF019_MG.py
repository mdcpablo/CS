import os
import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN_spherical as SN
import newGridXML2newGridObject
###############################################################################
def output_run_details(xs_file, mesh, L_max, comment, k):#, k, phi, psi, runtime_mg, iter_dict):
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
         f1.write('\nRESULTS\n')
         f1.write('k = %s \n' %k)
# -----------------------------------------------------------------------------
def run(xs_file, I=3, N=2, L_max=1, note=''):
    mat_dict = {'air': newGridXML2newGridObject.dict_to_object('xs/Air_'+xs_file, 10001),
                'heu': newGridXML2newGridObject.dict_to_object('xs/HEU19_'+xs_file, 92001),
                'gpht': newGridXML2newGridObject.dict_to_object('xs/Carbon12_'+xs_file, 12001)}

    air = SN.ZoneSpatialMesh('air', 0, 4.029, num_cells=I/3, log_option=False)
    heu = SN.ZoneSpatialMesh('heu', 4.029, 9.15, num_cells=I/3, log_option=False)
    gpht = SN.ZoneSpatialMesh('gpht', 9.15, 12.6, num_cells=I-2*(I/3), log_option=False)

    mesh = SN.GlobalMesh(mat_dict, [air,heu,gpht], N, 1)
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    bc = SN.BoundaryCondition(mesh, right='vacuum')

    k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'mg', mode='normal', L_max=L_max, tol=1e-8, max_its=50, k_exact=k_exact, DSA_opt=False)

    output_run_details(xs_file, mesh, L_max, 'MG', k)

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
k_exact = 1.00
#dofs = [400,600] #[100,200,400,600,800]
ps = [3,5]
sn = [32,64,128,256]
Is = [100,250,500,1000]

# needed for accuracy:
# 250 cells, P5, S128 
# -----------------------------------------------------------------------------
#dofs = [100,200,400,600,800]
dofs = [100]
mg_k, mg_k_error = run_dofs('HMF019_',dofs,'mg.xml',I=2,N=2,L_max=1,note='MG')
#mg_k, mg_k_error = run_dofs('HMF019_',dofs,'mg.xml',I=250,N=128,L_max=5,note='MG')






