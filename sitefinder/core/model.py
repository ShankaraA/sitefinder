import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from site_utils import *
from model_utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchnet.dataset import ListDataset
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel


'''

This file contains architectures used for SITEFINDER.

The current 'state of the art' achitecture for use is ConvNet4. This is a current work in 
progress to increase Recall in testing set. 


'''


class ConvNet(nn.Module):
    '''
    
    ConvNet: first attempt
    
    This model uses the encoder:decoder architecture and then a similar downsampling to get to a final N x 2 x 512 layer. 
    
    '''
    def __init__(self, input_channels=1, num_classes=2):
        super(ConvNet,self).__init__()
        
        # Params
        self.num_filters_1 = 32
        self.kernel_size_1 = (2, 8)
        self.stride_1 = (2, 8)
        
        self.num_filters_2 = 64
        self.kernel_size_2 = (2, 8)
        self.stride_2 = (2, 8)
        
        self.num_filters_3 = 128
        self.kernel_size_3 = (2, 2)
        self.stride_3 = (2, 4)
        
        self.num_classes = 2
        
        self.kernel_size_3up = (2, 4)
        
        self.kernel_dense1 = (1, 16)
        self.kernel_dense2 = (1, 2)
                                
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, self.num_filters_1, self.kernel_size_1,
                      stride=self.stride_1, padding=0),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU()
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(self.num_filters_1, self.num_filters_2, self.kernel_size_2, 
                      stride=self.stride_2, padding=0),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU()
        )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(self.num_filters_2, self.num_filters_3, self.kernel_size_3, 
                      stride=self.stride_3, padding=0),
            nn.BatchNorm2d(self.num_filters_3),
            nn.PReLU()
        )
        
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_3, self.num_filters_2, self.kernel_size_3up, 
                               stride=self.stride_3, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU()
        )
        
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_2, self.num_filters_1, self.kernel_size_2, 
                               stride=self.stride_2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU()
        )
        
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_1, self.num_classes, self.kernel_size_1, 
                               stride=self.stride_1, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
                
        self.flatten1 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
            
        self.flatten2 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
        
        self.flatten3 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense2, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
                
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encode1(x)
        out = self.encode2(out)
        out = self.encode3(out)
        out = self.decode3(out)
        out = self.decode2(out)
        out = self.decode1(out)
        out = self.flatten1(out)
        out = self.flatten2(out)
        out = self.flatten3(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        return F.log_softmax(out, dim=1)
    

class ConvNet2(nn.Module):
    '''
    
    ConvNet2
    
    This model has an identical architecture to ConvNet but incorporates dropout as a regularizaton technique at the downsampling layers. 
    
    '''
    def __init__(self, input_channels=1, num_classes=2):
        super(ConvNet2,self).__init__()
        
        # Params
        self.num_filters_1 = 32
        self.kernel_size_1 = (2, 8)
        self.stride_1 = (2, 8)
        
        self.num_filters_2 = 64
        self.kernel_size_2 = (2, 8)
        self.stride_2 = (2, 8)
        
        self.num_filters_3 = 128
        self.kernel_size_3 = (2, 2)
        self.stride_3 = (2, 4)
        
        self.num_classes = 2
        
        self.kernel_size_3up = (2, 4)
        
        self.kernel_dense1 = (1, 16)
        self.kernel_dense2 = (1, 2)
                                
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, self.num_filters_1, self.kernel_size_1,
                      stride=self.stride_1, padding=0),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU(),
            nn.Dropout(p=0.15,inplace=True)
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(self.num_filters_1, self.num_filters_2, self.kernel_size_2, 
                      stride=self.stride_2, padding=0),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU(),
            nn.Dropout(p=0.15,inplace=True)
        )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(self.num_filters_2, self.num_filters_3, self.kernel_size_3, 
                      stride=self.stride_3, padding=0),
            nn.BatchNorm2d(self.num_filters_3),
            nn.PReLU(),
            nn.Dropout(p=0.15,inplace=True)
        )
        
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_3, self.num_filters_2, self.kernel_size_3up, 
                               stride=self.stride_3, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU(),
            
            
        )
        
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_2, self.num_filters_1, self.kernel_size_2, 
                               stride=self.stride_2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU()
        )
        
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_1, self.num_classes, self.kernel_size_1, 
                               stride=self.stride_1, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
                
        self.flatten1 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=0.3, inplace=True)
        )
            
        self.flatten2 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=0.4, inplace=True)
        )
        
        self.flatten3 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense2, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=0.4, inplace=True)
        )
                
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encode1(x)
        out = self.encode2(out)
        out = self.encode3(out)
        out = self.decode3(out)
        out = self.decode2(out)
        out = self.decode1(out)
        out = self.flatten1(out)
        out = self.flatten2(out)
        out = self.flatten3(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        return F.log_softmax(out, dim=1)

class ConvNet4(nn.Module):
    '''

    ConvNet4
    
    This model uses an encoder:decoder architecture. A 2x512x512 image results from this. The tensor is that flattened and fed into a fully connected layer which is then used to output probabilities. 

    '''
    def __init__(self, input_channels=1, num_classes=2):
        super(ConvNet4,self).__init__()
        
        # Params
        self.num_filters_1 = 32
        self.kernel_size_1 = (2, 8)
        self.stride_1 = (2, 8)
        
        self.num_filters_2 = 64
        self.kernel_size_2 = (2, 8)
        self.stride_2 = (2, 8)
        
        self.num_filters_3 = 128
        self.kernel_size_3 = (2, 2)
        self.stride_3 = (2, 4)
        
        self.num_classes = 2
        
        self.kernel_size_3up = (2, 4)
        
        self.kernel_dense1 = (1, 16)
        self.kernel_dense2 = (1, 2)
                                
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, self.num_filters_1, self.kernel_size_1,
                      stride=self.stride_1, padding=0),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU(),
            nn.Dropout(p=0.2, inplace=True)
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(self.num_filters_1, self.num_filters_2, self.kernel_size_2, 
                      stride=self.stride_2, padding=0),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU(),
            nn.Dropout(p=0.3, inplace=True)
        )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(self.num_filters_2, self.num_filters_3, self.kernel_size_3, 
                      stride=self.stride_3, padding=0),
            nn.BatchNorm2d(self.num_filters_3),
            nn.PReLU(),
            nn.Dropout(p=0.3, inplace=True)
        )
        
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_3, self.num_filters_2, self.kernel_size_3up, 
                               stride=self.stride_3, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU(),
            nn.Dropout(p=0.25, inplace=True)
            
        )
        
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_2, self.num_filters_1, self.kernel_size_2, 
                               stride=self.stride_2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU(),
            nn.Dropout(p=0.25, inplace=True)
        )
        
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters_1, self.num_classes, self.kernel_size_1, 
                               stride=self.stride_1, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=0.4, inplace=True)
        )
        
        self.fc1 = nn.Linear(262144,512)
                 
                
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encode1(x)
        out = self.encode2(out)
        out = self.encode3(out)
        out = self.decode3(out)
        out = self.decode2(out)
        out = self.decode1(out)
        out = out.view(out.shape[0],2, -1)
        out = self.fc1(out)        
        m = nn.Dropout(p=0.4)
        out = m(out)
        
        return F.log_softmax(out, dim=1)
    
class ConvNet3(nn.Module):
    ''' 
    
    ConvNet3
    
    This model uses pixel shuffling to upsample following encoding of low resolution features. Following this, the 2x512x512 image is then downsampled using more convolutional layers to result in a 2 x 512 output. 
    
    
    '''


    def __init__(self, dropout = 0.9, input_channels=1, num_classes=2):
        super(ConvNet3,self).__init__()
        
        # Params
        self.num_filters_1 = 32
        self.kernel_size_1 = (4, 4)
        self.stride_1 = (2, 2)
        
        self.num_filters_2 = 64
        self.kernel_size_2 = (4, 4)
        self.stride_2 = (2, 2)
        
        self.num_filters_3 = 128
        self.kernel_size_3 = (4, 4)
        self.stride_3 = (2, 2)
        
        self.num_filters_4 = 256
        self.kernel_size_4 = (4, 4)
        self.stride_4 = (2, 2)   
        
        self.num_classes = 2
        
        self.kernel_size_3up = (2, 4)
        
        self.kernel_dense1 = (1, 16)
        self.kernel_dense2 = (1, 2)
                                
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, self.num_filters_1, self.kernel_size_1,
                      stride=self.stride_1, padding=1),
            nn.BatchNorm2d(self.num_filters_1),
            nn.PReLU()
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(self.num_filters_1, self.num_filters_2, self.kernel_size_2, 
                      stride=self.stride_2, padding=1),
            nn.BatchNorm2d(self.num_filters_2),
            nn.PReLU()
        )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(self.num_filters_2, self.num_filters_3, self.kernel_size_3, 
                      stride=self.stride_3, padding=1),
            nn.BatchNorm2d(self.num_filters_3),
            nn.PReLU()
        )
        
        self.encode4 = nn.Sequential(
            nn.Conv2d(self.num_filters_3, self.num_filters_4, self.kernel_size_4, 
                      stride=self.stride_4, padding=1),
            nn.BatchNorm2d(self.num_filters_4),
            nn.PReLU()
        )
        
        self.pixelshuffle = nn.PixelShuffle(16)
                    
        self.flatten1 = nn.Sequential(
            nn.Conv2d(1, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=dropout, inplace=True)
        )
            
        self.flatten2 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense1, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU(),
            nn.Dropout(p=dropout, inplace=True)
        )
        
        self.flatten3 = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, self.kernel_dense2, stride=(1,16), padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.PReLU()
        )
                
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encode1(x)
        out = self.encode2(out)
        out = self.encode3(out)
        out = self.encode4(out)
        
        out = self.pixelshuffle(out)

        out = self.flatten1(out)
        out = self.flatten2(out)
        out = self.flatten3(out)
        
        out = out.view(out.shape[0], out.shape[1], -1)
        
        return F.log_softmax(out, dim=1)