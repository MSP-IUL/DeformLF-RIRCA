#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:34:16 2024

@author: mzrdu
"""
import torch
import torch.nn as nn
import torch.nn.functional as functional
from Deformable import DeformableConv2d
from opt import opt
from Dataset import DatasetFromHdf5
class DepthNet(nn.Module):
    def __init__(self,opt):
        super(DepthNet, self).__init__()
        an2=opt.angular_out * opt.angular_out
        num_source=opt.angular_in * opt.angular_in
        self.conv1=DeformableConv2d(in_channels=num_source, out_channels=an2, kernel_size=3, stride=1, padding=1)
        self.conv2=DeformableConv2d(in_channels=num_source, out_channels=an2, kernel_size=5, stride=1, padding=2)
        self.x1=DeformableConv2d(in_channels=num_source, out_channels=16, kernel_size=3, stride=1, padding=1)
           
        self.x2=DeformableConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
            
        self.x3=DeformableConv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.x4=DeformableConv2d(in_channels=64, out_channels=an2, kernel_size=5, stride=1, padding=2)
        self.x5=DeformableConv2d(in_channels=an2, out_channels=an2, kernel_size=3, stride=1, padding=1)
        self.confuse=DeformableConv2d(in_channels=an2*3, out_channels=an2, kernel_size=1, stride=1, padding=0)
        
        self.lf_conv0=nn.Sequential(
            DeformableConv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )
        
    
        
        self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5,3,3), stride=(4,1,1), padding=(0,1,1)),#49-->12
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4,3,3), stride=(4,1,1), padding=(0,1,1)), #12-->3
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=49, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1)),#3-->1
            )      
        self.lr=nn.LeakyReLU(0.1)
            
        
    def forward(self, ind_source, img_source, LFI, opt):
        x=img_source
        x_m1=self.conv1(x)
        x_m1=self.lr(x_m1)
        x_m2=self.conv2(x)
        x_m2=self.lr(x_m2)
        x1=self.x1(img_source)
        x1=self.lr(x1)
        x2=self.x2(x1)
        x2=self.lr(x2)
        x3=self.x3(x2)
        x3=self.lr(x3)
        x4=self.x4(x3)
        x4=self.lr(x4)
        x5=self.x5(x4)
        x5=self.lr(x5)
        out_con=torch.cat([x_m1, x_m2, x5], dim=1)
       
        disp_target=self.confuse(out_con)
        N, num_source, h, w = img_source.shape
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out
        ind_source = torch.squeeze(ind_source)
        warp_img_input = img_source.view(N*num_source,1,h,w)
        warp_img_input =warp_img_input.repeat(an2, 1, 1, 1)
        grid = []
        for k_t in range(0,an2):
            for k_s in range(0,num_source):
                ind_s = ind_source[k_s].type_as(img_source)
                ind_t = torch.arange(an2)[k_t].type_as(img_source)
                ind_s_h = torch.floor(ind_s/an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t/an)
                ind_t_w = ind_t % an   
                disp = disp_target[:,k_t,:,:]
                XX = torch.arange(0,w).view(1,1,w).expand(N,h,w).type_as(img_source) #[N,h,w]
                YY = torch.arange(0,h).view(1,h,1).expand(N,h,w).type_as(img_source)
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w-1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h-1) - 1.0                
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm),dim=3)
                grid.append(grid_t)
        grid = torch.cat(grid, 0)
        warped_img = functional.grid_sample(warp_img_input,grid, align_corners=True).view(N,an2,num_source,h,w)
        
        #warped_img_view=torch.mean(warped_img_view, dim=2)
        warped_img_view=warped_img.contiguous().view(N*an2, num_source, h, w)
        depfeat=self.lf_conv0(warped_img_view)
       
        feat = torch.transpose(depfeat.view(N,an2,64,h,w),1,2) #[N,64,an2,h,w]
        res = self.lf_res_conv(feat) #[N,an2,1,h,w]
        
        SAI_out = warped_img[:,:,0,:,:] + torch.squeeze(res,2)
        
        
        return SAI_out
    
    
# dataset=DatasetFromHdf5(opt)
# model=DepthNet(opt)

# ind_source, input, label, LFI=dataset[0]
# # Convert ind_source to a PyTorch tensor
# ind_source = torch.from_numpy(ind_source)
# ind_source=ind_source.unsqueeze(0)
# input=input.unsqueeze(0)
# LFI=LFI.unsqueeze(0)
# output=model(ind_source, input, LFI, opt)
# print('Output Shape:', output.shape)



# # Calculate and print the number of parameters in millions
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('   Number of parameters: %.2fM' % (params / 1e6))    