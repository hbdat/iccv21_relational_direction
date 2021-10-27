# -*- coding: utf-8 -*- 
""" 
Created on Mon Jun 29 20:29:21 2020 
 
@author: badat 
""" 
import torch 
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
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
import pandas as pd 
import os 
 
 
class HICODataset(Dataset): 
    def __init__(self, partition, content, label_type = 'interaction'): 
        super(HICODataset, self).__init__()  
         
        self.grid_size = 7 
         
        self.partition = partition 
        self.hoi = pd.read_csv(NFS_path+'data/hico_20150920/hico_list_hoi.csv',header=None) 
        self.mask = np.zeros(len(self.hoi)) 
         
        self.label_type = label_type 
        self.partition = partition 
         
        assert self.partition in ['train_full','train_1A2B','train_1A','train_1A1B','test']  
        assert self.label_type in ['interaction','action','object','object_via_interact'] 
        self.content = content 
         
        self.i2a = np.argmax(self.content['Z_a'],axis=-1).astype(np.int32) 
        self.len_a = self.content['Z_a'].shape[-1] 
         
        self.i2o = np.argmax(self.content['Z_o'],axis=-1).astype(np.int32) 
        self.len_o = self.content['Z_o'].shape[-1] 
         
         
        self.samples = pd.read_csv(NFS_path+'data/hico_20150920/hico_unique_train_samples_1A2B.csv') 
        hico_1A = pd.read_csv(NFS_path+'data/hico_20150920/hico_1A.csv',header=None)[0].values 
        hico_2B = pd.read_csv(NFS_path+'data/hico_20150920/hico_2B.csv',header=None)[0].values 
         
        hico_2A = pd.read_csv(NFS_path+'data/hico_20150920/hico_2A.csv',header=None)[0].values 
        hico_1B = pd.read_csv(NFS_path+'data/hico_20150920/hico_1B.csv',header=None)[0].values 
         
        self.partition_1A = hico_1A 
        self.partition_2A = hico_2A 
        self.partition_1B = hico_1B 
        self.partition_2B = hico_2B 
         
        if partition == 'train_full': 
            self.samples = pd.read_csv(NFS_path+'data/hico_20150920/hico_unique_train_samples_1A2B.csv') 
            self.mask[:] = 1 
        elif partition == 'train_1A2B': 
            self.samples = pd.read_csv(NFS_path+'data/hico_20150920/hico_unique_train_samples_1A2B.csv') 
            ### occluding annotation to test zero-shot capacity ### 
            self.mask[hico_1A] = 1 
            self.mask[hico_2B] = 1 
            ### occluding annotation to test zero-shot capacity ### 
        elif partition == 'train_1A': 
            self.samples = pd.read_csv(NFS_path+'data/hico_20150920/hico_unique_train_samples_1A.csv') 
            ### occluding annotation to test zero-shot capacity ### 
            self.mask[hico_1A] = 1 
            ### occluding annotation to test zero-shot capacity ### 
        elif partition == 'test': 
            self.samples = pd.read_csv(NFS_path+'data/hico_20150920/hico_unique_test_samples.csv') 
            self.mask[:] = 1 
             
        self.samples = self.samples["file"].values 
        self.feature_dir = NFS_path+'data/hico_20150920/features/' 
 
    def __len__(self): 
        return len(self.samples) 
 
    def __getitem__(self, idx): 
        image_file = self.samples[idx] 
        feature_file = image_file.split(".")[0]+".pkl.gz" 
        feature_path = os.path.join(self.feature_dir,feature_file) 
         
        with gzip.open(feature_path, 'rb') as f: 
            content_load = pickle.load(f) 
         
        feature,label,image_file = content_load["feature"],content_load["label"],content_load["image_file"] 
         
        label = label * self.mask 
         
        pos_idx = np.where(label==1)[0] 
         
        if self.label_type == 'action': 
            label = np.ones(self.len_a)*-1 
            label[self.i2a[pos_idx]] = 1 
        elif self.label_type == 'object': 
            label = np.ones(self.len_o)*-1 
            label[self.i2o[pos_idx]] = 1 
        elif self.label_type == 'object_via_interact' and self.partition == 'test': 
            label = torch.ones(self.len_o)*-1 
            label[self.i2o[pos_idx]] = 1 
         
        return image_file,feature,label