import numpy as np
import sys, os, time, inspect
import xml.etree.ElementTree 
###################################################################################################
def make_dict_from_tree(element_tree):
    def internal_iter(tree, accum):
         if tree is None:
            return accum
         if tree.getchildren():
            accum[tree.tag] = {}
            for each in tree.getchildren():
                result = internal_iter(each, {})
                if each.tag in accum[tree.tag]:
                    if not isinstance(accum[tree.tag][each.tag], list):
                        accum[tree.tag][each.tag] = [
                            accum[tree.tag][each.tag]
                        ]
                    accum[tree.tag][each.tag].append(result[each.tag])
                else:
                    accum[tree.tag].update(result)
         else:
            accum[tree.tag] = tree.text
         return accum
    return internal_iter(element_tree, {})
###################################################################################################
def string_to_array(string, dtype=float):
    return np.array( filter(None, string.replace('\n',',').replace(' ',',').split(',')), dtype=dtype)
###################################################################################################
def xml_file_to_dict(xml_file):
    start = time.time()
    print "\nReading XML file... ",
    tree = xml.etree.ElementTree.parse(xml_file)
    print ("Done!\n> time to parse '"+xml_file.split('/')[-1]+"' file: %.2f s" %(time.time() - start))
    root = tree.getroot()
    return make_dict_from_tree(root)
###################################################################################################
class Data:
    def __init__(self, zaid):
        self.zaid = zaid
###################################################################################################
def dict_to_object(filePath, zaid):
    xs_dict = xml_file_to_dict(filePath)
    #print xs_dict.tags()
    data = Data(zaid)
    data.material = str(xs_dict['newGridXML']['materialList']).replace(' ','')
    data.crossSectionMTList = str(xs_dict['newGridXML']['crossSectionMTList'])
    data.num_ngrps = int(xs_dict['newGridXML']['numGroups'])
    data.num_grps = np.copy(data.num_ngrps)
    data.num_ggrps = 0
    data.nlgndr = int(xs_dict['newGridXML']['numMoments'])
    data.num_precursors = int(xs_dict['newGridXML']['numDelayedNeutronFlavors'])
    data.temperature = np.float(xs_dict['newGridXML']['temperature'])

    data.eBounds =  string_to_array(xs_dict['newGridXML']['grid']['values'])
    # make sure ALL your energies are in descending order
    #data.emid = 0.5*(data.eBounds[0:-1] + data.eBounds[1:])
    #data.de = (data.eBounds[0:-1] - data.eBounds[1:])  
    data.emid =  string_to_array(xs_dict['newGridXML']['energyMid']['values'])
    data.de =  string_to_array(xs_dict['newGridXML']['energydE']['values'])

    data.sigt =  string_to_array(xs_dict['newGridXML'][data.material]['MT_1'])
    data.p = []
    for moment in range(data.nlgndr):
        data.p.append(np.zeros((data.num_grps,data.num_grps)))
        for gp in range(data.num_grps):
            for g in range(data.num_grps):
                if 'toGroup_'+str(g) in xs_dict['newGridXML'][data.material]['MT_2519']['moment_'+str(moment)]['fromGroup_'+str(gp)]:
                    data.p[moment][gp,g] = np.float(xs_dict['newGridXML'][data.material]['MT_2519']['moment_'+str(moment)]['fromGroup_'+str(gp)]['toGroup_'+str(g)])

    data.sigf = np.zeros((data.num_grps,data.num_grps))
    data.sigf_tot = np.zeros(data.num_grps)

    if '2518' in data.crossSectionMTList:
        for gp in range(data.num_grps):
	        for g in range(data.num_grps):
	            if 'toGroup_'+str(g) in xs_dict['newGridXML'][data.material]['MT_2518']['fromGroup_'+str(gp)]:
	                data.sigf[gp,g] = np.float(xs_dict['newGridXML'][data.material]['MT_2518']['fromGroup_'+str(gp)]['toGroup_'+str(g)])
	                data.sigf_tot[gp] += data.sigf[gp,g]
        
    if '3519' in data.crossSectionMTList:
        data.numCoarseElements = int(xs_dict['newGridXML']['numCoarseElements'])
        data.coarse_p = []
        for moment in range(data.nlgndr):
            data.coarse_p.append(np.zeros((data.numCoarseElements,data.numCoarseElements)))
            for ep in range(data.numCoarseElements):
                for e in range(data.numCoarseElements):
                    if 'toElement_'+str(e) in xs_dict['newGridXML'][data.material]['MT_3519']['moment_'+str(moment)]['fromElement_'+str(ep)]:
                        data.coarse_p[moment][ep,e] = np.float(xs_dict['newGridXML'][data.material]['MT_3519']['moment_'+str(moment)]['fromElement_'+str(ep)]['toElement_'+str(e)])

    if '3518' in data.crossSectionMTList:
        data.numCoarseElements = int(xs_dict['newGridXML']['numCoarseElements'])
        data.coarse_sigf = np.zeros((data.numCoarseElements,data.numCoarseElements))
        data.coarse_sigf_tot = np.zeros(data.numCoarseElements)
        for ep in range(data.numCoarseElements):
            for e in range(data.numCoarseElements):
                if 'toElement_'+str(e) in xs_dict['newGridXML'][data.material]['MT_3518']['fromElement_'+str(ep)]:
                    data.coarse_sigf[ep,e] = np.float(xs_dict['newGridXML'][data.material]['MT_3518']['fromElement_'+str(ep)]['toElement_'+str(e)])
                    data.coarse_sigf_tot[ep] += data.coarse_sigf[ep,e]

        data.mapping = []
        for e in range(data.numCoarseElements):
            data.mapping.append(string_to_array(xs_dict['newGridXML']['map']['element_'+str(e)],dtype=int))
            
    if '2518' in data.crossSectionMTList:
        data.pdt_chid = np.zeros((data.num_precursors,data.num_grps))
        for d in range(data.num_precursors):
            for g in range(data.num_grps):
                    data.pdt_chid[d,:] = string_to_array(xs_dict['newGridXML'][data.material]['MT_2055']['delayedNeutronFlavor_'+str(d)])

    if '2518' in data.crossSectionMTList:
        data.decay_const =  string_to_array(xs_dict['newGridXML'][data.material]['MT_1054'])

        data.pdt_beta_fnubar = string_to_array(xs_dict['newGridXML'][data.material]['MT_455'])
        pdt_beta_fnubar = np.prod([data.pdt_beta_fnubar, data.sigf_tot], axis=0)

        if len(data.decay_const) != 0:
            data.chid = np.zeros((data.num_precursors, data.num_ngrps))
            data.beta_fnubar = np.zeros((data.num_precursors, data.num_ngrps))
            for d in range(data.num_precursors):
                data.chid[d] = data.pdt_chid[d]/np.sum(data.pdt_chid[d])
                data.beta_fnubar[d] = pdt_beta_fnubar * np.sum(data.pdt_chid[d])
        
    data.invSpgrp = string_to_array(xs_dict['newGridXML'][data.material]['MT_259'])
    data.spgrp =  np.reciprocal(data.invSpgrp)
    data.atwt = 0.60221409 #avogrados number
    data.siga = np.array([data.sigt[g] - sum(data.p[0][g]) for g in range(data.num_grps)])
    return data
###################################################################################################
