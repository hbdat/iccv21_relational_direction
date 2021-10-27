# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:12:48 2020

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
from torchvision.ops import roi_align
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score
from core.helper.helper_coordinate_func import attention_2_location
from core.helper.preprocessing_func import input_size_pad
#%%
def get_bbox_features(arr_feature_map,arr_pad_bboxes,img_size = 544.0):
#    tic = time.time()
    list_pad_bboxes = [pad_bboxes for pad_bboxes in arr_pad_bboxes]
#    print('[GPU-bboxes] Elapse {}'.format(time.time()-tic))
    
    
#    tic = time.time()
        
    spatial_scale = arr_feature_map.shape[-1]/img_size
    
    assert spatial_scale == 1/32.0
    
    bbox_features = roi_align(input=arr_feature_map, boxes=list_pad_bboxes, 
                              output_size=(7,7), spatial_scale=spatial_scale, 
                              sampling_ratio=-1) #[N*K,C,W,H]
#    print('[ROI] Elapse {}'.format(time.time()-tic))
    n_b = arr_pad_bboxes.shape[0]
    n_p = arr_pad_bboxes.shape[1]
    n_c = bbox_features.shape[1]
    n_w = n_h = bbox_features.shape[2]
    
    
    bbox_features = bbox_features.view(n_b,n_p,n_c,n_w,n_h)
    
    bbox_features = torch.mean(
            bbox_features.view(bbox_features.shape[0],bbox_features.shape[1],bbox_features.shape[2],-1),
            dim = -1)   #[N,K,C] <= [N,K,C,W*H]
    
    
#    bbox_features = bbox_features.permute(0,2,1) #[bfr] <== [brf]
    
    return bbox_features
#%%
class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
        
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
        
    def get_len(self):
        return len(self.df)
        
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
            
    def get_max(self,col):
        return np.max(self.df[col])
    
    def get_argmax(self,col):
        return np.argmax(self.df[col])
    
    def is_max(self,col):
        return self.df[col].iloc[-1] >= np.max(self.df[col])
#%%
def AP_score(gt,pred):
    npos = np.sum(gt)
    idx_sort = np.argsort(-pred)
    gt_sort = gt[idx_sort]
    
    nd = len(gt_sort)
    
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        if gt_sort[d] == 1:
            tp[d] = 1
        else:
            fp[d] = 1
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp/(fp+tp)
    
    ap = 0
    
    for t in np.arange(0,1.01,0.1):
        p = np.max(prec[rec>=t])
        
        if p is None:
            p = 0
        
        ap = ap + p/11
    
    return ap

def compute_AP(predictions,labels,type):
    print(type)
    num_class = predictions.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(predictions[:,idx_cls])
        label = np.squeeze(labels[:,idx_cls])
        if type=='realistic':
            mask = label != 0
        elif type == 'know object':
            mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1) 
        ap[idx_cls]=AP_score(binary_label,prediction[mask])#average_precision_score(binary_label,prediction[mask])#
    return ap    


def evaluate_mAP(dataloader, model, device, type = 'realistic'):
    all_preds = []
    all_labels =[]
    for i_batch, item in enumerate(dataloader):
        arr_file_name,arr_feature_map,arr_label = item[:3]
        with torch.no_grad():
            model.eval()
            features = arr_feature_map.to(device)
            
            out_package=model(features)
            preds = out_package['s'].cpu().numpy()
            labels = arr_label.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    return compute_AP(all_preds,all_labels,type),all_preds,all_labels

def evaluate_mAP_loc(dataloader, model, device, type = 'realistic'):
    all_preds = []
    all_coor_a = []
    all_coor_o = []
    all_labels =[]
    for i_batch, item in enumerate(dataloader):
        arr_file_name,arr_feature_map,arr_label,arr_size = item
        with torch.no_grad():
            model.eval()
            features = arr_feature_map.to(device)
            
            out_package=model(features)
            preds = out_package['s'].cpu().numpy()
            positions_o = out_package['positions_o'].cpu().numpy()
            positions_a = out_package['positions_a'].cpu().numpy()
            labels = arr_label.cpu().numpy()
            arr_size = arr_size.cpu().numpy()
            
            orginal_positions_o = np.ones(positions_o.shape)*-1
            orginal_positions_a = np.ones(positions_a.shape)*-1
            
            for j in range(positions_o.shape[0]):
                orginal_positions_o[j] = attention_2_location(coor = positions_o[j], grid_size = model.grid_size, test_input_size=input_size_pad, org_img_shape=arr_size[j])
                orginal_positions_a[j] = attention_2_location(coor = positions_a[j], grid_size = model.grid_size, test_input_size=input_size_pad, org_img_shape=arr_size[j])
                
                
            all_preds.append(preds)
            all_labels.append(labels)
            all_coor_a.append(orginal_positions_a)
            all_coor_o.append(orginal_positions_o)
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_coor_a = np.concatenate(all_coor_a)
    all_coor_o = np.concatenate(all_coor_o)
    
    
    return compute_AP(all_preds,all_labels,type),all_preds,all_labels
#%% compute F1 score
def compute_F1(predictions,labels,mode_F1):
    if mode_F1 == 'overall':
#        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)
    else:
        num_class = predictions.shape[1]
#        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls])
            if np.sum(label>0)==0:
                continue
            binary_label=np.clip(label,0,1)
            f1[idx_cls] = f1_score(binary_label,prediction)#AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label,prediction)
            r[idx_cls] = recall_score(binary_label,prediction)
    return f1,p,r

def evaluate_k(k, dataloader, model, device, predictions, labels, idx_labels = None):
    tic = time.time()
    if predictions is None:
        all_preds = []
        all_labels =[]
        for i_batch, (arr_file_name,arr_feature_map,arr_label) in enumerate(dataloader):
            with torch.no_grad():
                model.eval()
                features = arr_feature_map.to(device)
                
                out_package=model(features)
                preds = out_package['s'].cpu().numpy()
                labels = arr_label.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.copy(predictions)
        all_labels = np.copy(labels)
        
    ## binarize ##
    idx = np.argsort(all_preds,axis = 1)
    for i in range(all_preds.shape[0]):
        all_preds[i][idx[i][-k:]]=1
        all_preds[i][idx[i][:-k]]=0
    ## binarize ##
    
    if idx_labels is not None:
        all_preds = all_preds[:,idx_labels]
        all_labels = all_labels[:,idx_labels]
    
    assert all_preds.shape==all_labels.shape,'invalid shape'
#    print('Inference time {} n_samples {}'.format(time.clock()-tic,all_preds.shape[0]))
    return compute_F1(all_preds,all_labels,mode_F1='overall')
#%% 
def get_lr(optimizer): 
    lr = [] 
    for param_group in optimizer.param_groups: 
        lr.append(param_group['lr']) 
    return lr