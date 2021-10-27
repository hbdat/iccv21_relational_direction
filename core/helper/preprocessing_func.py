# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:46:30 2020

@author: badat
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
import pdb
import gzip
import json
import pandas as pd
import time
import torchvision.transforms.functional as F
#%%
input_size = 224
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

input_size_pad = 544
data_transforms_pad = transforms.Compose([
        transforms.Resize(input_size_pad),
        transforms.CenterCrop(input_size_pad),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_transforms_no_normalize = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])

data_transforms_original = transforms.Compose([
        transforms.ToTensor()
    ])


def get_padding(imsize):
    ### pad the image into square shape
    
    max_size = np.max(imsize)
    h_padding = (max_size - imsize[0]) / 2
    v_padding = (max_size - imsize[1]) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    
    return padding

def get_img_tensor(image_path):
    image = Image.open(image_path)
#    image = F.pad(image, get_padding(image))
    if image.mode == 'L':
        image=image.convert('RGB')
    image = data_transforms(image)
    return image

def get_img_tensor_no_normalize(image_path):
    image = Image.open(image_path)
#    image = F.pad(image, get_padding(image))
    if image.mode == 'L':
        image=image.convert('RGB')
    image = data_transforms_no_normalize(image)
    return image

def get_img_tensor_original(image_path):
    image = Image.open(image_path)
#    image = F.pad(image, get_padding(image))
    if image.mode == 'L':
        image=image.convert('RGB')
    image = data_transforms_original(image)
    return image

def get_img_tensor_pad(image_path):
    image = Image.open(image_path)
    
    if image.mode == 'L':
        image=image.convert('RGB')
    imsize = image.size
    image = F.pad(image, get_padding(imsize))
    
    img_tensor = np.asarray(image)
    
    assert img_tensor.shape[0] == img_tensor.shape[1]
    
    n_pad = np.sum(np.sum(np.abs(img_tensor),axis = -1) == 0)
    n_pixel = img_tensor.shape[0]**2
    
    ratio = n_pad/n_pixel
#    if ratio > 0.4:
#        print(ratio)
    
    image = data_transforms_pad(image)
    return image

def get_img_tensor_pad_original(image_path):
    image = Image.open(image_path)
    
    if image.mode == 'L':
        image=image.convert('RGB')
    imsize = image.size
    image = F.pad(image, get_padding(imsize))
    
    img_tensor = np.asarray(image)
    
    assert img_tensor.shape[0] == img_tensor.shape[1]
    
    n_pad = np.sum(np.sum(np.abs(img_tensor),axis = -1) == 0)
    n_pixel = img_tensor.shape[0]**2
    
    ratio = n_pad/n_pixel
#    if ratio > 0.4:
#        print(ratio)
    
    image = data_transforms_original(image)
    return image