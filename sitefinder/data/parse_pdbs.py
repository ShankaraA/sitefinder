import sys,os
import numpy as np
from os import listdir
from os.path import isfile, join
import Bio.PDB
from site_pdb import *

threshold = 512
data = np.zeros((25,512,512))
labels = np.zeros((25,512))
meta = np.zeros((25,2))

count = 1
num_pdb = 0

paths = ['./pdbs/pdb_list_sample/']
pdb_list = []

for path in paths:
    pdbs = [f for f in listdir(path) if isfile(join(path, f))]

    for pdb in pdbs:
        
        prot = ProteinSite(pdb[-8:-4], path+pdb, verbose=False)
        if prot.contact_map is not None:
            data[num_pdb,:,:] = prot.contact_map
            labels[num_pdb,:] = prot.labels
            meta[num_pdb,0] = int(prot.num_residues)
            meta[num_pdb,1] = int(prot.num_ligands)
            
            num_pdb+=1

            if num_pdb % 20 == 0:
                print('Loading no. ',num_pdb)
            
            pdb_list.append(prot.pdb)
            
        count+=1

print(num_pdb,' of ', count, ' have CM generation: ', 100 * num_pdb/count)
print("Complete!")
print("Saving...")

data = data[0:num_pdb,:,:]
labels = labels[0:num_pdb,:]
meta = meta[0:num_pdb,:]

np.save('./maps/maps_1',data)
np.save('./labels/labels_1', labels)
np.save('./lists/list_1',pdb_list)
np.save('./meta/meta_1',meta)




