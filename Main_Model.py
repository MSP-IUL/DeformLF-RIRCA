# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 21:04:22 2024

@author: MSP
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from DeformNet import DepthNet
from MIDeform import MI
from opt import opt
from Deformable import DeformableConv2d
from Dataset import DatasetFromHdf5
from Channel_attention import ConvChannelAttention
from DefUNET import DeformableEncoderDecoder
from Model_Utility import *
from RCB import DRCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):    
    def __init__(self, opt):
        super(Net, self).__init__()
        self.opt = opt  # Store opt as a class attribute
        an2 = opt.angular_out * opt.angular_out
        an = opt.angular_out
        self.an = opt.angular_out
        self.an2 = an * an
        #self.depth = DepthNet(opt)
        self.MI = MI(opt)
        ### Lenslet fuse ### To learn the LF angular feature in spatial domain as in SAI
        self.Fuse = make_FuseBlock(layer_num=2, a=an2, kernel_size=(3, 3))
        self.DRCA = DRCA()
    
    def forward(self, ind_source, img_source, LFI):
        opt = self.opt  # Access opt as a class attribute
        N, num_source, h, w = img_source.shape
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out
        
        # SAI_out = self.depth(ind_source, img_source, LFI, opt)
        MI_out = self.MI(LFI, img_source, opt)
        
        # Intermediate_out = MI_out + SAI_out
        
        fuse_out = self.Fuse(MI_out)
        out = self.DRCA(fuse_out)
        
        return out

class FuseBlock(nn.Module):
    def __init__(self, a, kernel_size, i):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
       
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out
        return out

def make_FuseBlock(layer_num, a, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(FuseBlock(a, kernel_size, i))
    return nn.Sequential(*layers)

# Load dataset and initialize model on device
dataset = DatasetFromHdf5(opt)
model = Net(opt).to(device)

# Prepare data, convert to tensors, and transfer to device
ind_source, input, label, LFI = dataset[0]
ind_source = torch.from_numpy(ind_source).unsqueeze(0).to(device)
input = input.unsqueeze(0).to(device)
LFI = LFI.unsqueeze(0).to(device)

# Run model forward pass
output = model(ind_source, input, LFI)
print('Output Shape:', output.shape)

# Calculate and print the number of parameters in millions
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %.2fM' % (params / 1e6))
