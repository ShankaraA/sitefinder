import pickle
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import Bio.PDB

'''

Site Utils

These util functions are used for site_pdb.py, the script that defines the class ProteinSite.
This is crucial for the data curation processing and not really needed during training. 

'''


def remove_nonligands(lig_dict, nonligands):
    '''
    
    Remove Nonligands. 
    
    Removes nonligands from ligand dictionary. Nonligands include ions, water, etc.

    Input: ligand dictonary - dict(ligand, AA residue), nonligands - list
    Output: ligand dictionary (edited)
    
    '''
    to_remove = []
    for nlig in nonligands:
        for key in lig_dict:
            if key[0] == nlig:
                to_remove.append(key)
                
    for key in to_remove:
        del lig_dict[key]
    return lig_dict


def resizeCM(input_arr, target_size=512, upper_tol=8, in_size=None):
    '''
    
    Author: Raphael Eugichi (slightly edited by Shankara Anand)
    
    This resizes and does padding for each pdb. Target size is the size of the contact map. 
    
    upper_tol: highest no. of residues that it will take in (520)
    in_size: if you already have size of the input - bypasses the size calculation
    
    
    '''
    
    upper_lim = target_size + upper_tol
    
    if not in_size:
        in_size = np.shape(input_arr)[0]
        assert in_size == np.shape(input_arr)[1]
        
    if in_size > upper_lim:
        return None 
    
    elif in_size <= target_size:
        if in_size % 2 == 0:
            pad = ((target_size - in_size)//2,)*2
        else:
            pad = ((target_size - (in_size-1))//2,(target_size - (in_size+1))//2)
        return np.pad(input_arr, pad, mode='constant'), (target_size, in_size), pad[0]
    
    else: 
        # if above target size, but within tolerance, center crop.
        temp_size = in_size
        temp_arr = input_arr
        if temp_size % 2 != 0:
            temp_arr = temp_arr[:temp_size-1, :temp_size-1] # truncate off the C-terminus if 
            temp_size -= 1
        trunc = (temp_size - target_size)//2
        return temp_arr[trunc:(temp_size-trunc), trunc:(temp_size-trunc)], (target_size, in_size), -trunc


    




