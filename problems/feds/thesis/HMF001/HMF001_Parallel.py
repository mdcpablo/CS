import os
import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import SN
import newGridXML2newGridObject
###############################################################################
def output_run_details(xs_file, mesh, L_max, comment):#, k, phi, psi, runtime_mg, iter_dict):
    outdir = xs_file.split('.')[0]    
    if not os.path.exists(outdir): 
        os.makedirs(outdir)  
    P = min(mesh.nlgndr, L_max)
    output_file = ('%s/%s_I%i_N%i_P%i.out' %(outdir,outdir,mesh.num_cells,mesh.num_angles,P))
    with open(output_file,'w') as f1:
         f1.write('xs_file = %s \n' %xs_file)
         f1.write('G = %s \n' %mesh.num_grps)
         f1.write('I = %s \n' %mesh.num_cells)
         f1.write('N = %s \n' %mesh.num_angles)
         f1.write('P = %s \n' %P)
# -----------------------------------------------------------------------------
def run(xs_file, I=2, N=2, L_max=1, note=''):
    mat_dict = {'hmf001': newGridXML2newGridObject.dict_to_object(xs_file, 92001)}

    heu_1 = SN.ZoneSpatialMesh('hmf001', 0, 8.7407, num_cells=I, log_option=False)

    mesh = SN.GlobalMesh(mat_dict, [heu_1], N, 1)
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    bc = SN.BoundaryCondition(mesh, right='vacuum')

    k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'mg', mode='normal', L_max=L_max, tol=1e-8, max_its=50, k_exact=k_exact, DSA_opt=False)

    output_run_details(xs_file, mesh, L_max, 'MG')

    return k, phi, psi, runtime_mg, iter_dict
# -----------------------------------------------------------------------------
def readfiles_dofs(xs_filename_left, dofs, xs_filename_right, I=2, N=2, L_max=1, note=''):
    k_list = []
    k_error_list = []
    for g in dofs:        
        outdir = xs_filename_left+str(g)+xs_filename_right.split('.')[0]     
        output_file = ('outputs/%s/%s_I%i_N%i_P%i.out' %(outdir,outdir,I,N,L_max))
        with open(output_file,'r') as f2:
            all_lines = f2.readlines()
        for line in all_lines:
            if 'k' in line.split('=')[0]:
                k = np.float(line.split('=')[1].split('\n')[0])
                print 
        k_list.append(np.copy(k))
        k_error_list.append(k-k_exact)
    return k_list, k_error_list
# -----------------------------------------------------------------------------
k_converged = 0.99520
k_exact = k_converged
#dofs = [400,600] #[100,200,400,600,800]
ps = [3,5]
sn = [32,64,128,256]
Is = [100,250,500,1000]

# needed for accuracy:
# 500 cells, P5, S128 
# -----------------------------------------------------------------------------
dofs = [200,400,600,800]
mg_k, mg_k_error = readfiles_dofs('HMF001_',dofs,'mg.xml',I=250,N=128,L_max=5,note='MG')
oldfeds_k, oldfeds_k_error = readfiles_dofs('HMF001_',dofs,'oldfeds_5EperCG.xml',I=250,N=128,L_max=5,note='old FEDS (5 elements per coarse group)')
newfeds_k, newfeds_k_error = readfiles_dofs('HMF001_',dofs,'newfeds.xml',I=250,N=128,L_max=5,note='new FEDS')
# -----------------------------------------------------------------------------
print mg_k
print oldfeds_k
print newfeds_k
# -----------------------------------------------------------------------------
mg_marker = 'bs'
oldfeds_marker = 'r^'
newfeds_marker = 'g*'
s = 10; alpha=0.6
# -----------------------------------------------------------------------------
plt.semilogx(dofs, mg_k, mg_marker, markersize=s, alpha=alpha, label='MG')
plt.semilogx(dofs, oldfeds_k, oldfeds_marker, markersize=s, alpha=alpha, label='old FEDS')
plt.semilogx(dofs, newfeds_k, newfeds_marker, markersize=1.5*s, alpha=alpha, label='new FEDS')
plt.xlabel(r'Number of Energy Groups')
plt.ylabel(r'$k$')
plt.xlim([1,1000])
plt.ylim([0.8,1.2])
plt.legend()
plt.show()
# -----------------------------------------------------------------------------
plt.loglog(dofs, map(abs, mg_k_error), mg_marker, markersize=s, alpha=alpha, label='MG')
plt.loglog(dofs, map(abs, oldfeds_k_error), oldfeds_marker, markersize=s, alpha=alpha, label='old FEDS')
plt.loglog(dofs, map(abs, newfeds_k_error), newfeds_marker, markersize=1.5*s, alpha=alpha, label='new FEDS')
plt.xlabel(r'Number of Energy Groups')
plt.ylabel(r'Error in $k$')
plt.xlim([1,1000])
plt.ylim([1e-8,1])
plt.legend()
plt.show()
# -----------------------------------------------------------------------------
plt.plot(dofs, mg_k_error, mg_marker, markersize=s, alpha=alpha, label='MG')
plt.plot(dofs, oldfeds_k_error, oldfeds_marker, markersize=s, alpha=alpha, label='old FEDS')
plt.plot(dofs, newfeds_k_error, newfeds_marker, markersize=1.5*s, alpha=alpha, label='new FEDS')
plt.xscale('log')
plt.yscale('symlog', linthreshy=1e-5)
plt.xlabel(r'Number of Energy Groups')
plt.ylabel(r'Error in $k$')
plt.xlim([1,1000])
plt.ylim([-1,1])
plt.legend()
plt.show()
# -----------------------------------------------------------------------------







