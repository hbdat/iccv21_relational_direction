# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:32:06 2020

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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.transform
from core.helper.preprocessing_func import get_img_tensor_no_normalize,input_size,get_img_tensor_original,get_img_tensor_pad_original,input_size_pad
import os
#%%
def visualize_attention(img_ids,A_o,A_a,labels,df_hoi,img_dir,save_path=None,prefix=''):          #A: [bkr]
     
    if (save_path is not None) and (not os.path.isdir(save_path)):
        os.mkdir(save_path)
    
    n = len(img_ids)        #pytorch only convert numerical value to tensor, string values are kept at their orginal types
    
    A_o = A_o.cpu().numpy()             #convert to numpy cpu format
    A_a = A_a.cpu().numpy()             #convert to numpy cpu format
    
    labels = labels.cpu().numpy()
    
    image_size = 14*16          #one side of the img
    r = A_o.shape[2]
    h = w =  int(np.sqrt(r))
    
    ret_idxs_pos = []
    for i in range(n):
        img_file=img_ids[i]
        img_name=img_file.split(".")[0]#.decode('utf-8')
        
#        visualize_dir = save_path+img_name+'/'
#       
        
        alpha_o = A_o[i]                #[kr]
        alpha_a = A_a[i]                #[kr]
        label = labels[i]           #[k]
        # Plot original image
        image = get_img_tensor_no_normalize(img_dir+img_file)
        image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 

        idxs_pos = np.where(label==1)[0]
        n_pos = len(idxs_pos)
        fig=plt.figure('',figsize=(10, 10))
        
        ret_idxs_pos.append(idxs_pos)
        
        for idx_r,idx_l in enumerate(idxs_pos):
            ax = plt.subplot(n_pos, 3, (idx_r*3)+1)
            plt.imshow(image)
            plt.axis('off')
            ax.set_title(img_name,{'fontsize': 10})
            
            ax = plt.subplot(n_pos, 3, (idx_r*3)+2)
            plt.imshow(image)
            alp_curr_o = alpha_o[idx_l,:].reshape(7,7)
            alp_img = skimage.transform.pyramid_expand(alp_curr_o, upscale=image_size/h, sigma=10,multichannel=False)
            plt.imshow(alp_img, alpha=0.7)
            plt.axis('off')
            ax.set_title("{}".format(df_hoi.iloc[idx_l]['obj']),{'fontsize': 10})
            
            ax = plt.subplot(n_pos, 3, (idx_r*3)+3)
            plt.imshow(image)
            alp_curr_a = alpha_a[idx_l,:].reshape(7,7)
            alp_img = skimage.transform.pyramid_expand(alp_curr_a, upscale=image_size/h, sigma=10,multichannel=False)
            plt.imshow(alp_img, alpha=0.7)
            plt.axis('off')
            ax.set_title("{}".format(df_hoi.iloc[idx_l]['act']),{'fontsize': 10})
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+prefix+img_file,dpi=500)
            plt.close()
        else:
            plt.show()
            
        return ret_idxs_pos
#%%
img_dir = os.path.join(NFS_path,'data/hico_20160224_det/images/')

def visualize_loc_origin_size(img_name, pred_det,df_hoi,save_path=None,prefix=''):          #A: [bkr]
       
    if (save_path is not None) and (not os.path.isdir(save_path)):
        os.mkdir(save_path)
    
    labels = list(pred_det.keys())

    # Plot original image
    image = get_img_tensor_original(img_dir+img_name+'.jpg')
    image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 
    
    image_pad = get_img_tensor_pad_original(img_dir+img_name+'.jpg')
    image_pad = image_pad.permute(1,2,0) #[224,244,3] <== [3,224,224]     
    fig=plt.figure('',figsize=(10, 10))
    
    n_pos = len(labels)
    ext = [0.0, input_size[1], 0.00, input_size[0]]
    for idx_l,l in enumerate(labels):
        ax = plt.subplot(n_pos, 3, (idx_l*3)+1)
        plt.imshow(image)
        plt.axis('off')
        ax.set_title(img_name,{'fontsize': 10})
        
        ax = plt.subplot(n_pos, 3, (idx_l*3)+2)
        pos_a_l = pred_det[l]['pos_img_a']
        plt.scatter(pos_a_l[0], pos_a_l[1],zorder=1,marker='o',color='r')
        plt.imshow(image, zorder=0, extent=ext)
        #plt.axis('off')
        ax.set_title("{}".format(df_hoi.iloc[l-1]['act']),{'fontsize': 10})
        
        ax = plt.subplot(n_pos, 3, (idx_l*3)+3)
        pos_o_l = pred_det[l]['pos_img_o']
        plt.scatter(pos_o_l[0], pos_o_l[1],zorder=1,marker='o',color='r')
        plt.imshow(image, zorder=0, extent=ext)
        #plt.axis('off')
        ax.set_title("{}".format(df_hoi.iloc[l-1]['obj']),{'fontsize': 10})
        
        print("img size: {} pos_img_o: {} pos_img_a {}".format(image.shape,pos_o_l,pos_a_l))
        print("pos_grid_o: {} pos_grid_a {}".format(pred_det[l]['pos_grid_o'],pred_det[l]['pos_grid_a']))
        
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+prefix++img_name+'.jpg',dpi=500)
        plt.close()
    else:
        plt.show()
        
    
    fig=plt.figure('',figsize=(10, 10))
    
    n_pos = len(labels)
    
    for idx_l,l in enumerate(labels):
        ax = plt.subplot(n_pos, 3, (idx_l*3)+1)
        plt.imshow(image_pad)
        plt.axis('off')
        ax.set_title(img_name,{'fontsize': 10})
        
        ax = plt.subplot(n_pos, 3, (idx_l*3)+2)
        plt.imshow(image_pad)
        alpha_a_l = pred_det[l]['A_a']
        w = h = int(np.sqrt(alpha_a_l.shape[-1]))
        alpha_a_l = alpha_a_l.reshape(w,h)
        alp_img = skimage.transform.pyramid_expand(alpha_a_l, upscale=input_size_pad/h, sigma=10,multichannel=False)
        plt.imshow(alp_img, alpha=0.7)
        plt.axis('off')
        ax.set_title("{}".format(df_hoi.iloc[l-1]['act']),{'fontsize': 10})
        
        ax = plt.subplot(n_pos, 3, (idx_l*3)+3)
        plt.imshow(image_pad)
        alpha_o_l = pred_det[l]['A_o']
        w = h = int(np.sqrt(alpha_o_l.shape[-1]))
        alpha_o_l = alpha_o_l.reshape(w,h)
        alp_img = skimage.transform.pyramid_expand(alpha_o_l, upscale=input_size_pad/h, sigma=10,multichannel=False)
        plt.imshow(alp_img, alpha=0.7)
        plt.axis('off')
        ax.set_title("{}".format(df_hoi.iloc[l-1]['obj']),{'fontsize': 10})
        
        print(image_pad.shape)
        
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+prefix++img_name+'.jpg',dpi=500)
        plt.close()
    else:
        plt.show()
#%%
cmap = plt.get_cmap("tab20b")
bbox_color = cmap(0)
def add_bbox(bbox,ax):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    
    box_w = x2 - x1
    box_h = y2 - y1

    color = bbox_color
    # Create a Rectangle patch
    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
    # Add the bbox to the plot
    ax.add_patch(bbox)    
    
def visualize_bbox_origin_size(img_ids,Bbox_o,Bbox_a,labels,df_hoi,img_dir,save_path=None,prefix=''):          #A: [bkr]
     
    if (save_path is not None) and (not os.path.isdir(save_path)):
        os.mkdir(save_path)
    
    n = len(img_ids)        #pytorch only convert numerical value to tensor, string values are kept at their orginal types
    
    labels = labels.cpu().numpy()
        
    ret_idxs_pos = []
    for i in range(n):
        img_file=img_ids[i]
        img_name=img_file.split(".")[0]#.decode('utf-8')

        bbox_o = Bbox_o[i]                #[kr]
        bbox_a = Bbox_a[i]                #[kr]
        label = labels[i]           #[k]
        # Plot original image
        image = get_img_tensor_no_normalize(img_dir+img_file)
        image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 

        idxs_pos = np.where(label==1)[0]
        n_pos = len(idxs_pos)
        fig=plt.figure('',figsize=(10, 10))
        
        ret_idxs_pos.append(idxs_pos)
        
        for idx_r,idx_l in enumerate(idxs_pos):
            ax = plt.subplot(n_pos, 3, (idx_r*3)+1)
            plt.imshow(image)
            plt.axis('off')
            ax.set_title(img_name,{'fontsize': 10})
            
            ax = plt.subplot(n_pos, 3, (idx_r*3)+2)
            plt.imshow(image)
            bbox_l_o = bbox_o[idx_l]
            add_bbox(bbox_l_o,ax)
            plt.axis('off')
            ax.set_title("{}".format(df_hoi.iloc[idx_l]['obj']),{'fontsize': 10})
            
            ax = plt.subplot(n_pos, 3, (idx_r*3)+3)
            plt.imshow(image)
            bbox_l_a = bbox_a[idx_l]
            add_bbox(bbox_l_a,ax)
            plt.axis('off')
            ax.set_title("{}".format(df_hoi.iloc[idx_l]['act']),{'fontsize': 10})
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+prefix+img_file,dpi=500)
            plt.close()
        else:
            plt.show()
            
        return ret_idxs_pos