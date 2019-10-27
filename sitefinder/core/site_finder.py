import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt

from site_utils import *
from model_utils import *
from data_utils import *
from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchnet.dataset import ListDataset
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel

device, num_workers, pin_memory, dtype = check_gpu()

batch_size = 500
num_classes = 2
print_every = 5
split = 0.8

print('Loading data...')
data = np.load('./datasets/dataset_400_maps.npy')
labels = np.load('./datasets/dataset_400_smoothlabels.npy')
weights = get_frequency(labels)
labels[labels > 0] = 1

train_loader, val_loader = get_loaders_from_numpy(data, labels, batch_size, split=split)

      
# Define Loss Function
weight = torch.from_numpy(weights).to(device=device, dtype=dtype)
criterion = CrossEntropyLoss2d(size_average=False, weight=weight)

def train(model, optimizer, lr_updater, results, epochs=5):
    
    for e in range(epochs):
        
        lr_updater.step()
        
        for t, (x,y) in enumerate(train_loader):
            model.train()
            
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            
            scores = model.forward(x)

            loss = criterion(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch %d/%d, loss = %.4f' % (e+1, epochs, loss.item()))
        train_metrics = check_metrics(train_loader, model, 'Training  ')
        val_metrics = check_metrics(val_loader, model, 'Validation')
        print()
        results[int(e+1)] = [loss.item(), train_metrics, val_metrics]
    return results
  
        
def train_model(model, model_name, hyperparams, device, epochs):
    '''
    
    Train Model
    
    This is a generic function to call the model's training function. 
    
    '''

    print('Beginning Training for: ', model_name)
    print('------------------------------------')
    
    results = {}
    
    if torch.cuda.device_count() > 1: 
        print("Using ", torch.cuda.device_count(), " GPUs.")
        print('------------------------------------')
        model = DataParallel(model)
        
    model = model.to(device=device)
    
    optimizer = optim.Adam(model.parameters(), betas=hyperparams['betas'], lr=hyperparams['learning_rate'], weight_decay=hyperparams['L2_reg'])
    
    lr_updater = lr_scheduler.StepLR(optimizer, hyperparams['lr_decay_epochs'], hyperparams['lr_decay'])
    
    results = train(model, optimizer, lr_updater, results, epochs=epochs)
    
    plot_results(results, model_name ,save=True)
    np.save(model_name, results)
    
    return results


print('--------TRAINING--------')
epochs = 320
name = 'cnn2_full_xSmooth1'

hyperparams = {}

hyperparams['betas'] = (0.9, 0.99)
hyperparams['L2_reg'] = 1e-3

hyperparams['learning_rate'] = 0.001
hyperparams['lr_decay_epochs'] = 80
hyperparams['lr_decay'] = 0.8

model = ConvNet2()

hyp = name + '_hyper.npy'
mod = name + '.pt'

results = train_model(model, name, hyperparams, device, epochs)
np.save(hyp, hyperparams)
model.save_state_dict(name+'.pt')
      














