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
def run(xs, g, I=2, N=2, L_max=1, note=''):
    mat_dict = {'vacuum':   newGridXML2newGridObject.dict_to_object(('xs/%s/Air_%i.xml' %(xs,g)), 92001),
                'heu':      newGridXML2newGridObject.dict_to_object(('xs/%s/HEU19_%i.xml' %(xs,g)), 92001),
                'graphite': newGridXML2newGridObject.dict_to_object(('xs/%s/Carbon12_%i.xml' %(xs,g)), 92001)}

    I = int(10*int((I+5)/10))
    vacuum = SN.ZoneSpatialMesh('vacuum', 0, 4.029, num_cells=I/10, log_option=False)
    heu = SN.ZoneSpatialMesh('heu', 4.029, 9.15, num_cells=I*7/10, log_option=False)
    graphite = SN.ZoneSpatialMesh('graphite', 9.15, 12.6, num_cells=I*2/10, log_option=False)

    mesh = SN.GlobalMesh(mat_dict, [vacuum, heu, graphite], N, 5, 'sphere')
    mesh.print_energies()
    mesh.print_angles()
    mesh.print_space(v=0)

    bc = SN.BoundaryCondition(mesh, right='vacuum')
     
    recomp_F = 4
    recomp_S = [2,4,4,4,4,16,16,16,16,16]

    k, phi, psi, runtime_mg, iter_dict = SN.power_iterations(mesh, bc, 'k', 'cs', mode='normal', L_max=L_max, tol=1e-5, max_its=1000, k_exact=k_exact, DSA_opt=False, recomp_F=recomp_F, recomp_S=recomp_S)

    comment = 'cs, recomp_F='+str(recomp_F)+', recomp_S='+str(recomp_S)
    #output_run_details(xs_file, mesh, L_max, comment, k, runtime_mg, iter_dict)

    return k, phi, psi, runtime_mg, iter_dictHMF
# -----------------------------------------------------------------------------
def run_dofs(xs, dofs, I=2, N=2, L_max=1, note=''):
    k_list = []
    k_error_list = []
    for g in dofs:        
        k, phi, psi, runtime_mg, iter_dict = run(xs, g, I=I, N=N, L_max=L_max, note=note)
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
dofs = [250]
#dofs = [100]
mg_k, mg_k_error = run_dofs('vfeds_nocg',dofs,I=100,N=256,L_max=8,note='vfeds_nocg')


