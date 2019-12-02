import os

from_dir = '/home/pablo/barnfire/dissertation_problems'

for g in [100,200,400,600,800]:
    g = int(g)
    os.system('cp %s/hmf085_3_%img/BarnfireXS_SteelyKnife_%i.xml xs/Steel_%img.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%img/BarnfireXS_UnpolutedAir_%i.xml xs/Air_%img.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%img/BarnfireXS_HEU85_%i.xml xs/HEU85_%img.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%ioldfeds_5EperCG/BarnfireXS_SteelyKnife_%i.xml xs/Steel_%ioldfeds_5EperCG.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%ioldfeds_5EperCG/BarnfireXS_UnpolutedAir_%i.xml xs/Air_%ioldfeds_5EperCG.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%ioldfeds_5EperCG/BarnfireXS_HEU85_%i.xml xs/HEU85_%ioldfeds_5EperCG.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%inewfeds/BarnfireXS_SteelyKnife_%i.xml xs/Steel_%inewfeds.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%inewfeds/BarnfireXS_UnpolutedAir_%i.xml xs/Air_%inewfeds.xml' %(from_dir,g,g,g))
    os.system('cp %s/hmf085_3_%inewfeds/BarnfireXS_HEU85_%i.xml xs/HEU85_%inewfeds.xml' %(from_dir,g,g,g))
