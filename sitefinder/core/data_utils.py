import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt

from model_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchnet.dataset import ListDataset
import torch.optim.lr_scheduler as lr_scheduler


def get_frequency(labels):
    '''
    
    Get Frequency
    
    This returns the binary class weight distribution of a given set of samples. 
    
    '''
    class_0, class_1 = 0, 0
    
    for i in tqdm(range(labels.shape[0])): 
        lab = labels[i,:]
        
        class_0 += len(lab[lab == 0])
        class_1 += len(lab[lab == 1])
        
    return np.array([class_1/(class_0+class_1), class_0/(class_0+class_1)])

def filter_by_length(files, threshold = 500, labels_path = './data/parsed/labels/'):
    '''
    
    Filter by Length
    
    Reads in all pdb files, returns the actual length  of each pdb and a list of 
    protein files that are above a certain threshold length. This was used in
    results for SITEFINDER to first test SITEFINDER's model on proteins above 400
    amino acids in length.
    
    '''
    
    residue_lengths = []
    protein_files = []
    
    for f in files: 
        val = np.load(labels_path+f)
        res_length = len(val[val > -1])
        
        residue_lengths.append(res_length)
        
        if res_length > threshold:
            protein_files.append(f)
        
    return residue_lengths, protein_files

def get_loaders_from_numpy(data, labels, batch_size, split = 0.8, verbose=True):
    '''
    
    Get Loaders from Numpy
    
    This takes in numpy arrays of data and labels, batch size, and the split rate to return
    training and validation dataloaders. It also apprpriately assigns num_workers and pin memory
    based on GPU availability.
    
    '''
    
    assert (data.shape[0] == labels.shape[0]), 'No. samples must agree!'
    
    device, num_workers, pin_memory, dtype = check_gpu(verbose=False)
    
    split_num = int(split * data.shape[0])
    
    x_train = torch.from_numpy(data[:split_num,:,:])
    y_train = torch.from_numpy(labels[:split_num,:]).to(dtype=torch.long)
    x_val = torch.from_numpy(data[split_num:,:,:])
    y_val = torch.from_numpy(labels[split_num:,:]).to(dtype=torch.long)
    
    
    if verbose:
        print('X train: ', x_train.shape)
        print('Y train: ', y_train.shape)
        print('X val: ', x_val.shape)
        print('Y val: ', y_val.shape)
    

    train_dataset = TensorDataset(x_train,y_train)
    val_dataset = TensorDataset(x_val,y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader


def generate_output(input_map, input_label, model):
    '''
    
    Generate Output
    
    This takes in an input map and input label, both as numpy arrays, and outputs the predicted
    label. The requirement for adding an input_label is to force padding that exists when 
    first processing the protein.
    
    '''
    protein_map = torch.from_numpy(input_map).to(dtype=torch.float)
    
    predicted_label = model(protein_map)
    predicted_label = np.asarray(predicted_label.argmax(dim=1))
    predicted_label[input_label < 0] = -1
    
    output = np.zeros((2,512))
    output[0,:] = input_label
    output[1,:] = predicted_label
    
    return output
    
def load_func(line, map_path = './data/parsed/maps/', labels_path = './data/parsed/labels/'):
    '''
    
    Load Func
    
    This function is used if loading the entire PDB dataset (not the 400 
    length filtered one). Because the entire dataset cannot fit to memory, 
    a dataloader that loads contact maps while training is requied. 
    
    '''
    src = torch.from_numpy(np.load(map_path+line)).type(torch.FloatTensor)
    
    target = np.load(labels_path+line)
    target[target > 0] = 1
    target = torch.from_numpy(target)
    
    return {'src': src, 'target': target}

def batchify(batch):
    '''
    
    Batchify
    
    How the Dataloader creates batches. See 'load_func'.
    
    '''
    src_list = []
    target_list = []
    
    for value in batch:
        src_list.append(value['src'])
        target_list.append(value['target'])
    
    batch_src = torch.stack(src_list).view(-1,512,512)
    batch_target = torch.stack(target_list).view(-1,512)

    return (batch_src, batch_target)

# How to use load func and batchify. 
# train_dataset = ListDataset(data_files[:split_num], load_func)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=batchify)

# val_dataset = ListDataset(data_files[split_num:], load_func)
# val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=batchify)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    