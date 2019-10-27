from builtins import range
from builtins import object
from site_utils import *
import numpy as np
import Bio.PDB
import os.path


class ProteinSite(object):
    '''
    
    Protein Site class built for SITEFINDER. This class takes in a protein data bank (PDB) file, does
    the necessary parsing to find what ligands are bound using metadata found in PDB files. It also 
    generates contact maps and labels that are later used for SITEFINDER. 

    This uses site_utils for helper functions. 

    '''
    def __init__(self, pdb_id, pdb_file, site='Ligand', verbose=False):
        
        # Nonligands
        nonligands = ['SO4','CL','NH4','CA','PO4','ZN','NA','MG','SF4','FE', 'CD', 'CO', 'NI']
        threshold = 520
        map_size = 512
        
        # Load PDB into ProteinSite Object
        if verbose:
            print('####### LOADING ',pdb_id,' ##################') 
            print('# File: ', pdb_file)
        self.pdb, self.site = pdb_id, site
        
        # Get Protein Residues and Ligands
        try: 
            self.residues, self.res_dict = ProteinSite.getRes(pdb_file)
            self.num_residues = len(self.residues)
            
            self.ligands = remove_nonligands(ProteinSite.ligandDict(pdb_file), nonligands)
            self.num_ligands = len(self.ligands)

            # Get unique ligands
            ligs = set()
            
            for lig in list(self.ligands.keys()):
                ligs.add(lig[0])
            if ligs == set():
                ligs = None
            self.unique_ligands = ligs
            
        except:
            self.num_residues = 999
            self.num_ligands = 0
            self.unique_ligands = 0
        
        # Build Contact Map and Labels
        self.contact_map = None
        in_size = self.num_residues
        self.labels = None
        
        if self.num_residues <= threshold and self.num_ligands > 0:
            try: 
                cMap = ProteinSite.getContactMap(pdb_file)
                self.contact_map, (map_size, in_size), self.map_mod = resizeCM(cMap)
                self.labels = ProteinSite.getLabels(self, map_size)
            except:
                print("Could not generate contact map.")
                       
        # Verbose 
        if verbose:
            print('# Number of residues: ', self.num_residues)
            print('# Number of ligands: ', self.num_ligands)
            print('# Ligands: ', self.unique_ligands)
            print('# In size: ',in_size,' --> Target size: ',map_size)
            print('#########################################\n')
                    
    def ligandDict(pdb_file):
        '''
        
        This function returns a dictionary of ligand sites in each protein. 

        '''
        site_dict = {}
        lig_dict = {}
        
        # Get Ligand Dictionary
        with open(pdb_file) as f:
            for line in f:
                if "REMARK 800" in line:
                    if line.split()[-1] != 'NULL':
                        if "SITE_IDENTIFIER" in line: 
                            site_id = line.split()[-1]
                        
                        # Can modify here to include catalytic sites later
                        if "SITE_DESCRIPTION" in line and ('BINDING SITE FOR' or 'binding site for' in line):
                            # Edge case where AA # is 4 digits
                            if len(line.split()[-1]) > 4: 
                                val = line.split()[-1]
                                ligand = (line.split()[-2], val[0], val[1:])
                            else:
                                ligand = line.split()[-3:]
                            site_dict[site_id] = ligand

                if "SITE" in line and line.split()[2] in site_dict:
                    lig = tuple(site_dict[line.split()[2]])
                    if lig not in lig_dict:
                        entries = []
                    else: 
                        entries = lig_dict[lig]
                    
                    # Edge case for Res #s with 4 digits for correct parsing
                    line = ' '.join([val[0] + ' ' + val[1:] if len(val)>4 else val for val in line.split()])
                    
                    for res, chain, res_num in zip(*[iter(line.split()[4:])]*3):
                        entries.append((res,chain,int(res_num)))
                        lig_dict[lig] = entries   
        return lig_dict
    
    
    def getRes(pdb_file):
        '''
        
        This returns a list of residues in the protein in sequence order along with a dictionary of residues
        to their PDB numbering. 

        Residues are undefined heteroatoms (i.e. res_id[3][0] == ' ')

        '''
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_file[-8:-4], pdb_file)
        res_list = []
        res_no = 1
        res_dict = {}

        for res in structure.get_residues():
            res_id = res.get_full_id()

            # Only scrape for actual residues
            if res_id[3][0] == ' ':
                tup = (res.get_resname(),res_id[2],res_id[3][1])
                res_list.append(tup)
                res_dict[tup] = res_no

                # Iterate Res Number Counter
                res_no += 1
                
        return res_list, res_dict
    
    def getContactMap(pdb):
        '''

        Author: Raphael Eugichi

        This function generates contact maps for an input protein. 


        '''
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb[:-4], pdb)
        A = []
        for model in structure:
            for chain in model:
                for res in chain:
                    try:
                        A.append(np.asarray(res['CA'].get_coord())) # C-alpha coordinates are extracted in residue order.
                    except:
                        continue
        return distance_matrix(A,A)
    
    def getLabels(self,map_size):
        '''
        
        This function gets labels, i.e. what residues in a protein are involved in drug binding. 


        '''
        labels = np.zeros(map_size)
        
        # Correct for Padding
        if self.map_mod > 0:
            labels[:self.map_mod] = -1
            labels[-self.map_mod:] = -1
        
        # Add Labels for Ligand Binding Site (0 for not involved, 1 for functional site residue)
        for key in self.ligands:
            residues = self.ligands[key]
            for res in residues:
                if res in self.residues:
                    adjusted_index = self.res_dict[res] + self.map_mod - 1
                    if adjusted_index < map_size:
                        labels[adjusted_index]+=1
        return labels
        
        
