import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchnet.dataset import ListDataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel.data_parallel import DataParallel


class CrossEntropyLoss2d(nn.Module):
    '''
    
    Author: Raphael Eugichi
    
    Advice for which loss function to use was given by Raphel Eugichi, 
    who used protein contact maps to create a domain segmenter for the course project CS230. 
    
    
    '''
    def __init__(self, weight=None, size_average=True, ignore_index=-1):

        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)

    
def check_gpu(verbose=True):
    '''
    
    This function checks if a GPU is available. If so, it sets the number
    of workers to 0 and pins the data loaded to memory for quicker access by GPU.
    
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_workers = 0
        pin_memory = True
        if verbose:
            print("Using GPU.")
    else:
        device = torch.device('cpu')
        num_workers = 8
        pin_memory = False

    return device, num_workers, pin_memory, torch.float32


def check_metrics(loader, model, set_name):
    '''
    
    Check Metrics
    
    This computes evaluation metrics for a given loader using current model state. 
    This computes precision, recall, accuracy, and F1 score. Values of -1 are given if 
    precision and recall do not exist. 
    
    
    '''
    true_pos, false_pos, true_neg, false_neg, num_samples = 0, 0, 0, 0, 0
    device, num_workers, pin_memory, dtype = check_gpu(verbose=False)
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            _, preds = scores.max(1)
            
            pos_indices, neg_indices = (y==1), (y==0)    
            
            true_pos += (preds[pos_indices] == 1).sum()
            false_pos += (preds[neg_indices] == 1).sum()
            true_neg += (preds[neg_indices] == 0).sum()
            false_neg += (preds[pos_indices] == 0).sum()
            num_samples += int(y.shape[0] * y.shape[1]) - int((y == -1).sum())
            
        # Compute Precision
        if float(true_pos + false_pos) == 0:
            precision = -1
        else:
            precision = float(true_pos) / float(true_pos + false_pos)
        
        # Compute Recall
        if float(true_pos + false_neg) == 0:
            recall = -1
        else:
            recall = float(true_pos) / float(true_pos + false_neg)
            
        acc = float(true_pos + true_neg) / float(num_samples)
        f1_score = float(2 * true_pos) / float(2 * true_pos + false_pos + false_neg)
        
        metrics = {'accuracy': acc, 'precision': precision, 'recall': recall, 'F1 score': f1_score}
        print('%s ==> Precision: %.4f | Recall: %.4f | F1: %.2f | Accuracy: %.3f' % (set_name, metrics['precision'], metrics['recall'], metrics['F1 score'], metrics['accuracy']))
        
        return metrics


def plot_results(results, file_name='default', save=False):
    '''
    
    Plot Results

    General plotting function for viewing metrics during training. 
    
    
    '''
    L = sorted(results.items())
    e, Z = zip(*L)
    
    loss, train_precision, train_recall, train_f1, train_acc = [], [], [], [], []
    val_precision, val_recall, val_f1, val_acc = [], [], [], []
    
    val_flag = False
    
    for i,z in enumerate(Z):
        loss.append(z[0])
        train_metrics, val_metrics = z[1], z[2]
        
        train_precision.append(train_metrics['precision'])
        train_recall.append(train_metrics['recall'])
        train_f1.append(train_metrics['F1 score'])
        train_acc.append(train_metrics['accuracy'])
        
        if val_metrics:
            val_flag = True
            val_precision.append(val_metrics['precision'])
            val_recall.append(val_metrics['recall'])
            val_f1.append(val_metrics['F1 score'])
            val_acc.append(val_metrics['accuracy'])        
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
                
    ax1.plot(e, loss, label='Train Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    
    ax2.plot(e, train_precision, label='Train')
    ax2.legend(loc='upper left')
    ax2.set_title('Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Precision')
    
    ax3.plot(e, train_recall, label='Train')
    ax3.legend(loc='upper left')
    ax3.set_title('Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')

    ax4.plot(e, train_f1, label='Train')
    ax4.legend(loc='upper left')
    ax4.set_title('F1')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    
    if val_flag:
        ax2.plot(e, val_precision, label='Val')
        ax3.plot(e, val_recall, label='Val')
        ax4.plot(e, val_f1, label='Val')
    
    fig.set_size_inches(12, 8, forward=True)
    plt.tight_layout()
    
    if save:
        plt.savefig(file_name+'.png',dpi=300)
    plt.show()
    