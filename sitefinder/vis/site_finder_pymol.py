
import sys
import pymol
from pymol import cmd
from site_pymol_utils import *
import numpy as np

'''

This script is run directly in PYMOL. 

Open PYMOL. Then cd into the directory with this script. Ensure that you have the output file generated from
"Test Model.ipynb". 

"run site_finder_pymol.py" in the PYMOL console.

"load PDB ID"

"truth PDB ID" -- this colors the pdb with ground truth labels provided

"site PDB ID" -- this colors the pdb with predicted labels


'''

def load(pdb):
	'''
	
	This funciton loads a pdb into the PYMOL environment. 


	'''
	cmd.fetch(str(pdb))
	
	cmd.remove('solvent')
	cmd.color('green')

	file_name = str(pdb) + '.pdb'
	cmd.save(file_name)

def truth(pdb):
	'''

	This function takes in the loaded pdb and out files from SITEFINDER (see to_pymol.zip), and colors the protein accordingly. 


	'''
	pdb_map, res_shift = getMap(str(pdb) + '.pdb')
	ligands, start_index = parsePDB(str(pdb) + '.pdb')
	pdb_labels = np.load(str(pdb) + '_output.npy')

	pdb_labels = pdb_labels[[0],:]

	cmd.bg_color('white')

	cmd.color('yellow')
	cmd.set('cartoon_transparency','0.2')

	print(res_shift)
	for i in range(pdb_labels.shape[1]):
		if pdb_labels[0,i] > 0:
			new_index = i - res_shift + int(start_index)
			color_command = 'resi ' + str(new_index)
			cmd.color('red', color_command)

	for lig in ligands:
		color_command = 'resn ' + lig
		cmd.color('blue', color_command)

def site(pdb):
	'''

	This colors the protein using SITEFINDER's predicted ligand binding sites. 

	'''
	pdb_map, res_shift = getMap(str(pdb) + '.pdb')
	ligands, start_index = parsePDB(str(pdb) + '.pdb')
	pdb_labels = np.load(str(pdb) + '_output.npy')

	pdb_labels = pdb_labels[[1],:]

	cmd.bg_color('white')

	cmd.color('yellow')
	cmd.set('cartoon_transparency','0.2')

	print(res_shift)
	for i in range(pdb_labels.shape[1]):
		if pdb_labels[0,i] > 0:
			new_index = i - res_shift + int(start_index)
			color_command = 'resi ' + str(new_index)
			cmd.color('red', color_command)

	for lig in ligands:
		color_command = 'resn ' + lig
		cmd.color('blue', color_command)

cmd.extend('load', load)
cmd.extend('site', site)
cmd.extend('truth', truth)