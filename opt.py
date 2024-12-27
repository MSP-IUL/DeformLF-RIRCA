#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:18:30 2024

@author: mzrdu
"""
import argparse
import numpy as np
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)
        
parser =argparse.ArgumentParser(description="PyTorch Light Field Hybrid SR")
#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=500, help="Learning rate decay every n epochs") # 学习率下降间隔数 每500次将学习率调整为lr*reduce
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=96, help="Training patch size")
parser.add_argument("--channel_size", type=int, default=64, help="channels size")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_cp", type=int, default=10, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=5, help="Number of epochs for saving loss figure")
parser.add_argument("--dataset", type=str, default="Real", help="Dataset for training")
# parser.add_argument("--dataset_path", type=str, default="./LFData/train_SIG.h5")
parser.add_argument("--dataset_path", type=str, default="train_SIG.H5")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field")
parser.add_argument("--angular_in", type=int, default=2, help="angular number of the sparse light field [AngIn x AngIn]")
parser.add_argument("--test_dataset", type=str, default="Real", help="dataset for testing")
parser.add_argument("--data_path", type=str, default="Test_Reflective.h5")
parser.add_argument("--save_img", type=str, default='save_img', help="save image or not 1/0")
parser.add_argument("--crop", type=int, default=0, help="crop the image into patches when out of memory")
parser.add_argument("--model_path", type=str, default="pretrained_model/model_epoch_2990.pth", help="pretrained model path")
opt  = parser.parse_args(args=[])
print(opt)