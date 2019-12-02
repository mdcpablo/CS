import os,sys

prob_lowercase = 'hmf019'
prob_uppercase = 'HMF019'
xs = 'feds_nocg'
from_mats = ['UnpolutedAir', 'Carbon12', 'HEU19']
to_mats = ['Air', 'Carbon12', 'HEU19']

new_path = os.path.join(os.path.join(prob_uppercase,'xs'),xs)
os.system('mkdir %s' %new_path)

G = [50,100,150,200,250,300,350,400,500,600]
for g in G:   
    for i in range(len(from_mats)):
        os.system('cp /home/pablo/barnfire/dissertation_problems/%s/%s/%i/BarnfireXS_%s_%i.xml %s/xs/%s/%s_%i.xml' %(prob_lowercase,xs,g,from_mats[i],g,prob_uppercase,xs,to_mats[i],g))
