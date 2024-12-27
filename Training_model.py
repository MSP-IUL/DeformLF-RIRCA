#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:38:15 2024

@author: mzrdu
"""
import torch
from opt import opt
from Dataset import DatasetFromHdf5
import random
import numpy as np
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader
import torch.optim as optim
import math
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Main_Model import Net
#--------------------------------------------------------------------------#

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
#--------------------------------------------------------------------------#
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

opt.num_source = opt.angular_in * opt.angular_in
model_dir = 'model_{}_S{}'.format(opt.dataset, opt.num_source)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#--------------------------------------------------------------------------#
# Data loader
print('===> Loading datasets')
#dataset_path = join('LFData', 'train_{}.h5'.format(opt.dataset))
train_set = DatasetFromHdf5(opt)
train_loader = DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=True)
print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))
#--------------------------------------------------------------------------#
# Build model
print("building net")

model = Net(opt).to(device)
#-------------------------------------------------------------------------#
# optimizer and loss logger
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
losslogger = defaultdict(list)
#------------------------------------------------------------------------#    
# optionally resume from a checkpoint
if opt.resume_epoch:
    resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
    # resume_path = join(model_dir, 'model_epoch.pth')
    if os.path.isfile(resume_path):
        print("==>loading checkpoint 'epoch{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losslogger = checkpoint['losslogger']
    else:
        print("==> no model found at 'epoch{}'".format(opt.resume_epoch))


#------------------------------------------------------------------------#
# loss
def reconstruction_loss(X,Y):
# L1 Charbonnier loss
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss

def train(epoch):

    model.train()
    # scheduler.step()
    loss_count = 0.

    for k in range(10):
         for i, batch in enumerate(train_loader, 1):
            ind_source, input, label, LFI = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            pred_views= model(ind_source, input, LFI)
            loss = reconstruction_loss(pred_views, label)
            loss_count += loss.item()
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scheduler.step()
    losslogger['epoch'].append(epoch)
    losslogger['loss'].append(loss_count/len(train_loader))
    return loss_count/len(train_loader)

# #-------------------------------------------------------------------------#
print('==>training')
min=10
for epoch in range(opt.resume_epoch+1, 3000):
    loss = train(epoch)
    with open("./loss.txt", "a+") as f:
        f.write(str(epoch))
        f.write("\t")
        f.write(str(loss))
        f.write("\t")
        tim = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f.write(str(tim))
        f.write("\n")

#     checkpoint
    if epoch % opt.num_cp == 0:
        model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))
        state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
        torch.save(state, model_save_path)
        if min > loss:
            min = loss
            print("update")
            print(min)
            print(epoch)
            print("checkpoint saved to {}".format(model_save_path))

    # loss snapshot
    if epoch % opt.num_snapshot == 0:
        plt.figure()
        plt.title('loss')
        plt.plot(losslogger['epoch'],losslogger['loss'])
        plt.savefig(model_dir+".jpg")
        plt.close()
        