import os

for g in [100,200,400,600,800]:
    g = int(g)
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%img/BarnfireXS_Carbon12_%i.xml HMF019/xs/Carbon12_%img.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%img/BarnfireXS_UnpolutedAir_%i.xml HMF019/xs/Air_%img.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%img/BarnfireXS_HEU19_%i.xml HMF019/xs/HEU19_%img.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%ioldfeds_5EperCG/BarnfireXS_Carbon12_%i.xml HMF019/xs/Carbon12_%ioldfeds_5EperCG.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%ioldfeds_5EperCG/BarnfireXS_UnpolutedAir_%i.xml HMF019/xs/Air_%ioldfeds_5EperCG.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%ioldfeds_5EperCG/BarnfireXS_HEU19_%i.xml HMF019/xs/HEU19_%ioldfeds_5EperCG.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%inewfeds/BarnfireXS_Carbon12_%i.xml HMF019/xs/Carbon12_%inewfeds.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%inewfeds/BarnfireXS_UnpolutedAir_%i.xml HMF019/xs/Air_%inewfeds.xml' %(g,g,g))
    os.system('cp /home/pablo/barnfire/dissertation_problems/hmf019_%inewfeds/BarnfireXS_HEU19_%i.xml HMF019/xs/HEU19_%inewfeds.xml' %(g,g,g))
