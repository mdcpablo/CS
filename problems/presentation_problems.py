import sys
sys.path.append('/home/pablo/CS/src')
import numpy as np
import matplotlib.pyplot as plt
import time
import PowerIterations as PI
import newGridXML2newGridObject
###############################################################################
def runprint(eigenval_name, results):
    print 'Final '+eigenval_name+' = '+str(results[0])+ '  (runtime = '+str(results[2])+')'
###############################################################################
k_results = []; a_results = []
###############################################################################
# FEDS problems
#data = newGridXML2newGridObject.dict_to_object('xs/heu20_5000_invSpgrp_feds_300.xml', 92001)
#data = newGridXML2newGridObject.dict_to_object('xs/heu20_300_feds_cs_30.xml', 92001)
#data = newGridXML2newGridObject.dict_to_object('xs/fudge_HEU20_1000.xml', 92001)
data = newGridXML2newGridObject.dict_to_object('newGrid_hmf001_cs_25.xml', 92235)

k_results.append(PI.PI(data, 'k', 'mg', talk=True));     runprint('k', k_results[-1])
#a_results.append(PI.PI(data, 'alpha', 'mg', talk=True)); runprint('alpha', a_results[-1]) 
###############################################################################
# CS problems
true_false_recompute_list = []
for i in range(1000):
    if i%2 == 0: #recompute S_e->g every 2 iterations
        S_eg = True
    else:
        S_eg = False

    if i%4 == 0: #recompute F_e->g every 4 iterations
        F_eg = True
    else:
        F_eg = False
    true_false_recompute_list.append([S_eg,F_eg]) 

