# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:03:04 2020

@author: badat
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models.resnet as models
from PIL import Image
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
import pdb
import gzip
import json
from core.helper.preprocessing_func import get_img_tensor_pad
#%%
#import pdb
#%%
idx_GPU = 7
is_save = True
#%%
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#%%
img_dir = os.path.join(NFS_path,'data/Visual_Genome/VG_100K/')
#pdb.set_trace()
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%%

model_ref = models.resnet152(pretrained=True)
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()

for param in model_f.parameters():
    param.requires_grad = False
#%%
class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir , dic_path, transform=None):
        self.dic_path = dic_path
        with open(dic_path, 'rb') as f:
            self.anno = pickle.load(f)
        
        self.image_files = np.array(list(self.anno.keys()))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.img_dir,image_file+".jpg")
        idxs_label = self.anno[image_file]
        label = np.ones(6643)*-1
        label[idxs_label] = 1
        image = get_img_tensor_pad(image_path)
        return image,label,image_file

#%%
input_size = 224
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
#%%
def sanity_check(dict_a,dict_b):
    assert dict_a.keys() == dict_b.keys()
    for k in dict_a.keys():
        if type(dict_a[k]) == np.ndarray:
            assert (dict_a[k] == dict_b[k]).all()
        else:
            assert dict_a[k] == dict_b[k]
#%%   
dic_path = './data/Visual_Genome/dic_anno_train_1A2B.pkl'
HICODataset = CustomedDataset(img_dir , dic_path, data_transforms)
dataset_loader = torch.utils.data.DataLoader(HICODataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

with torch.no_grad():
    for i_batch, (imgs,labels,image_files) in enumerate(dataset_loader):
        print(i_batch)
        imgs=imgs.to(device)
        features = model_f(imgs)
        for i_f in range(features.size(0)):
            feature = features[i_f].cpu().numpy()
            
            label = labels[i_f].cpu().numpy()
            
            image_file = image_files[i_f]
            save_file = image_file.split(".")[0]+".pkl.gz"
            
            content = {'feature':feature,'label':label,'image_file':image_file} 
            with gzip.open(NFS_path+"data/Visual_Genome/features_pad/{}".format(save_file), 'wb') as f:
                pickle.dump(content,f)
            
            ### need sanity check ### << load it back and compare the result
            with gzip.open(NFS_path+"data/Visual_Genome/features_pad/{}".format(save_file), 'rb') as f:
                content_load = pickle.load(f)
            sanity_check( content, content_load)
#%%
dic_path = './data/Visual_Genome/dic_anno_test.pkl'
HICODataset = CustomedDataset(img_dir , dic_path, data_transforms)
dataset_loader = torch.utils.data.DataLoader(HICODataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

with torch.no_grad():
    for i_batch, (imgs,labels,image_files) in enumerate(dataset_loader):
        print(i_batch)
        imgs=imgs.to(device)
        features = model_f(imgs)
        for i_f in range(features.size(0)):
            feature = features[i_f].cpu().numpy()
            
            label = labels[i_f].cpu().numpy()
            
            image_file = image_files[i_f]
            save_file = image_file.split(".")[0]+".pkl.gz"
            
            content = {'feature':feature,'label':label,'image_file':image_file} 
            with gzip.open(NFS_path+"data/Visual_Genome/features_pad/{}".format(save_file), 'wb') as f:
                pickle.dump(content,f)
            
            ### need sanity check ### << load it back and compare the result
            with gzip.open(NFS_path+"data/Visual_Genome/features_pad/{}".format(save_file), 'rb') as f:
                content_load = pickle.load(f)
            sanity_check( content, content_load)
