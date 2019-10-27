import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

'''

This script takes arrays parsed straight form the pdb loader and saves individual numpy arrays for each label 
and map for easy reading into SITEFINDER. 

'''
def write_out(maps, labels, names, output_dir='./parsed/'):

    for idx, pdb in enumerate(names):
        np.save(output_dir + 'maps/' + pdb, maps[[idx],:,:])
        np.save(output_dir + 'labels/' + pdb, labels[[idx],:])
        
map_dir = './maps/maps_'
labels_dir = './labels/labels_'
list_dir = './lists/list_'

files = ['1.npy']

for file in tqdm(files):
    maps = np.load(map_dir+file)
    labels = np.load(labels_dir+file)
    ls = np.load(list_dir+file)
    tqdm(write_out(maps,labels,ls))