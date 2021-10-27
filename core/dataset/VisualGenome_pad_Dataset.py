# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:27:13 2020

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


class VisualGenome_pad_Dataset(Dataset):
    def __init__(self, partition, label_type = 'interaction'):
        super(VisualGenome_pad_Dataset, self).__init__() 
        
        self.grid_size = 17
        
        self.partition = partition
        self.hoi = pd.read_csv(NFS_path+'data/Visual_Genome/VG_list_hoi.csv',header=None)
        self.mask = np.zeros(len(self.hoi))
        
        self.label_type = label_type
        self.partition = partition
        
        assert self.partition in ['train_1A2B','train_1A','test'] 
        assert self.label_type in ['interaction']
#        self.content = content
        
        VG_1A = pd.read_csv(NFS_path+'data/Visual_Genome/VG_1A.csv',header=None)[0].values
        VG_2B = pd.read_csv(NFS_path+'data/Visual_Genome/VG_2B.csv',header=None)[0].values
        
        VG_2A = pd.read_csv(NFS_path+'data/Visual_Genome/VG_2A.csv',header=None)[0].values
        VG_1B = pd.read_csv(NFS_path+'data/Visual_Genome/VG_1B.csv',header=None)[0].values
        
        self.partition_1A = VG_1A
        self.partition_2A = VG_2A
        self.partition_1B = VG_1B
        self.partition_2B = VG_2B
        
        ## filter out non-testable label ##
        self.filter = np.concatenate([VG_1A,VG_1B,VG_2A,VG_2B],axis=0)
        assert len(self.filter) == len(self.hoi) 
#        common_class = pd.read_csv(NFS_path+'data/Visual_Genome/common_class.csv',header=None)[0].values
        ## filter out non-testable label ##
        
#        self.i2a = np.argmax(self.content['Z_a'],axis=-1).astype(np.int32)
#        self.i2a = self.i2s[self.filter]    #filter
#        self.len_a = self.content['Z_a'].shape[-1]
#        
#        self.i2o = np.argmax(self.content['Z_o'],axis=-1).astype(np.int32)
#        self.i2o = self.i2o[self.filter]    #filter
#        self.len_o = self.content['Z_o'].shape[-1]
        
        
        if partition == 'train_full':
            dic_path = './data/Visual_Genome/dic_anno_train_1A2B.pkl'
            with open(dic_path, 'rb') as f:
                self.anno = pickle.load(f)
            self.samples = np.array(list(self.anno.keys()))
            
            self.mask[:] = 1
        elif partition == 'train_1A2B':
            dic_path = './data/Visual_Genome/dic_anno_train_1A2B.pkl'
            with open(dic_path, 'rb') as f:
                self.anno = pickle.load(f)
            self.samples = np.array(list(self.anno.keys()))
            
            ### occluding annotation to test zero-shot capacity ###
            self.mask[VG_1A] = 1
            self.mask[VG_2B] = 1
            ### occluding annotation to test zero-shot capacity ###
        elif partition == 'train_1A':
            dic_path = './data/Visual_Genome/dic_anno_train_1A.pkl'
            with open(dic_path, 'rb') as f:
                self.anno = pickle.load(f)
            self.samples = np.array(list(self.anno.keys()))
            
            ### occluding annotation to test zero-shot capacity ###
            self.mask[VG_1A] = 1
            ### occluding annotation to test zero-shot capacity ###
        elif partition == 'test':
            dic_path = './data/Visual_Genome/dic_anno_test.pkl'
            with open(dic_path, 'rb') as f:
                self.anno = pickle.load(f)
            self.samples = np.array(list(self.anno.keys()))
            
            self.mask[:] = 1
            
        self.feature_dir = NFS_path+'data/Visual_Genome/features_pad/'

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
        
#        label = label[self.filter] #filter


#        pos_idx = np.where(label==1)[0]
#        
#        if self.label_type == 'action':
#            label = np.ones(self.len_a)*-1
#            label[self.i2a[pos_idx]] = 1
#        elif self.label_type == 'object':
#            label = np.ones(self.len_o)*-1
#            label[self.i2o[pos_idx]] = 1
#        elif self.label_type == 'object_via_interact' and self.partition == 'test':
#            label = torch.ones(self.len_o)*-1
#            label[self.i2o[pos_idx]] = 1
        
        return image_file,feature,label