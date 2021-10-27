# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:02:39 2020

@author: badat
"""

import os
import argparse
import h5py
from tqdm import tqdm
#import matplotlib.pyplot as plt
from multiprocessing import Pool

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

import core.helper.utils.io as io
#%%
def compute_ap(precision,recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0,1.1,0.1): # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall>=t]
        if selected_p.size==0:
            p = 0
        else:
            p = np.max(selected_p)   
        ap += p/11.
    
    return ap


def compute_pr(y_true,y_score,npos):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall

def load_gt_dets(proc_dir,global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir,'anno_list.json')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets

def is_inside(loc,bbox,verbose=False):
    x1,y1 = loc
    x1_,y1_,x2_,y2_ = bbox
    
    if x1 >= x1_ and x1 <= x2_:
        if y1 >= y1_ and y1 <= y2_:
            return True

    return False

def match_loc(pred_det,gt_dets,type_interact,type):
    is_match = False
    for i,gt_det in enumerate(gt_dets):
        
        if type == 'single':
            match_h = is_inside(pred_det['human_loc'],gt_det['human_box'])
        elif type == 'or' or type == 'or_random':
            match_h = is_inside(pred_det['human_loc'],gt_det['human_box']) or is_inside(pred_det['cross_human_loc'],gt_det['human_box'])
        elif type == 'cross':
            match_h = is_inside(pred_det['cross_human_loc'],gt_det['human_box'])
        
        
        if type == 'single':
            match_o = is_inside(pred_det['object_loc'],gt_det['object_box'])
        elif type == 'or' or type == 'or_random':
            match_o = is_inside(pred_det['object_loc'],gt_det['object_box']) or is_inside(pred_det['cross_object_loc'],gt_det['object_box'])
        elif type == 'cross':
            match_o = is_inside(pred_det['cross_object_loc'],gt_det['object_box'])
        
            
        if type_interact == 'hoi':
            if match_h and match_o:
                is_match = True
                break
        elif type_interact == 'human':
            if match_h:
                is_match = True
                break
        elif type_interact == 'object':
            if match_o:
                is_match = True
                break
        #remaining_gt_dets.append(gt_det)

    return is_match

def eval_hoi(hoi_id,img_names,gt_dets,pred_dets,verbose = False,type='single',skip_absent=False):
    if verbose:
        print(f'Evaluating hoi_id: {hoi_id} ...')
    
    y_true_hoi = []
    y_true_human = []
    y_true_object = []
    
    y_score = []
#    det_id = []
    npos = 0
    
    for idx,global_id in enumerate(img_names):
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            if skip_absent:
                continue
            candidate_gt_dets = []
        npos += int(len(candidate_gt_dets)>0)
        
        if hoi_id not in pred_dets[global_id]:
            continue
        
        pred_det = {
            'human_loc': pred_dets[global_id][hoi_id]['pos_img_a'],
            'object_loc': pred_dets[global_id][hoi_id]['pos_img_o'],
            'score': pred_dets[global_id][hoi_id]['score']
        }
        
        if type in ['or','cross','or_random']:
            pred_det['cross_human_loc'] = pred_dets[global_id][hoi_id]['cross_pos_img_a']
            pred_det['cross_object_loc'] = pred_dets[global_id][hoi_id]['cross_pos_img_o']
        
        
        is_match_hoi = match_loc(pred_det,candidate_gt_dets,'hoi',type)
        is_match_human = match_loc(pred_det,candidate_gt_dets,'human',type)
        is_match_object = match_loc(pred_det,candidate_gt_dets,'object',type)
        
        y_true_hoi.append(is_match_hoi)
        y_true_human.append(is_match_human)
        y_true_object.append(is_match_object)
        
        y_score.append(pred_det['score'])
    
    ## has no correct retrieval ##
    ap_hoi = 0
    if len(y_true_hoi) != 0:
        # Compute PR
        precision_hoi,recall_hoi = compute_pr(y_true_hoi,y_score,npos)
        
        # Compute AP
        ap_hoi = compute_ap(precision_hoi,recall_hoi)
    
    ap_human = 0
    if len(y_true_human) != 0:
        # Compute PR
        precision_human,recall_human = compute_pr(y_true_human,y_score,npos)
        
        # Compute AP
        ap_human = compute_ap(precision_human,recall_human)
        
    ap_object = 0  
    if len(y_true_object) != 0:
        # Compute PR
        precision_object,recall_object = compute_pr(y_true_object,y_score,npos)
        
        # Compute AP
        ap_object = compute_ap(precision_object,recall_object)
    
    if verbose:
        print(f'AP_hoi:{ap_hoi} AP_human:{ap_human} AP_object:{ap_object}')
    
    # Save AP data
#    ap_data = {
#        'y_true': y_true,
#        'y_score': y_score,
#        'det_id': det_id,
#        'npos': npos,
#        'ap': ap,
#    }
#    np.save(
#        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
#        ap_data)

    return (ap_hoi,ap_human,ap_object,hoi_id)

# ### new for iccv -- GT-Known Loc ###
# def eval_acc_loc_hoi(hoi_id,img_names,gt_dets,pred_dets,verbose = False,type='single',metric='Top_Loc'):
#     if verbose:
#         print(f'Evaluating hoi_id: {hoi_id} ...')
    
#     y_true_hoi = []
#     y_true_human = []
#     y_true_object = []
    
#     y_score = []
# #    det_id = []
#     npos = 0
    
#     for idx,global_id in enumerate(img_names):
#         if hoi_id in gt_dets[global_id]:
#             candidate_gt_dets = gt_dets[global_id][hoi_id]
#         else:
#             if metric == 'GT_Known_Loc':
#                 continue
#             candidate_gt_dets = []
#         npos += int(len(candidate_gt_dets)>0)
        
#         if hoi_id not in pred_dets[global_id]:
#             continue
        
#         pred_det = {
#             'human_loc': pred_dets[global_id][hoi_id]['pos_img_a'],
#             'object_loc': pred_dets[global_id][hoi_id]['pos_img_o'],
#             'score': pred_dets[global_id][hoi_id]['score']
#         }
        
#         if type in ['or','cross','or_random']:
#             pred_det['cross_human_loc'] = pred_dets[global_id][hoi_id]['cross_pos_img_a']
#             pred_det['cross_object_loc'] = pred_dets[global_id][hoi_id]['cross_pos_img_o']
        
        
#         is_match_hoi = match_loc(pred_det,candidate_gt_dets,'hoi',type)
#         is_match_human = match_loc(pred_det,candidate_gt_dets,'human',type)
#         is_match_object = match_loc(pred_det,candidate_gt_dets,'object',type)
        
#         y_true_hoi.append(is_match_hoi)
#         y_true_human.append(is_match_human)
#         y_true_object.append(is_match_object)
        
#         y_score.append(pred_det['score'])
    
#     ## has no correct retrieval ##
#     acc_hoi = 0
#     if len(y_true_hoi) != 0:
#         # Compute PR
#         # precision_hoi,recall_hoi = compute_pr(y_true_hoi,y_score,npos)
        
#         # Compute AP
#         acc_hoi = sum(y_true_hoi)/npos#compute_ap(precision_hoi,recall_hoi)
    
#     acc_human = 0
#     if len(y_true_human) != 0:
#         # Compute PR
#         # precision_human,recall_human = compute_pr(y_true_human,y_score,npos)
        
#         # Compute AP
#         acc_human = sum(y_true_human)/npos#compute_ap(precision_human,recall_human)
        
#     acc_object = 0  
#     if len(y_true_object) != 0:
#         # Compute PR
#         # precision_object,recall_object = compute_pr(y_true_object,y_score,npos)
        
#         # Compute AP
#         acc_object = sum(y_true_object)/npos#compute_ap(precision_object,recall_object)
    
#     if verbose:
#         print(f'AP_hoi:{acc_hoi} AP_human:{acc_human} AP_object:{acc_object}')
    
#     # Save AP data
# #    ap_data = {
# #        'y_true': y_true,
# #        'y_score': y_score,
# #        'det_id': det_id,
# #        'npos': npos,
# #        'ap': ap,
# #    }
# #    np.save(
# #        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
# #        ap_data)

#     return (acc_hoi,acc_human,acc_object,hoi_id)

#%%
class LocEvaluator_HICODet:
    def __init__(self,dataloader,k=10,verbose = False, num_processes = 50):
        
        self.gt_dir = NFS_path+'/data/hico_20160224_det/json/'
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.num_processes = num_processes
        self.k = k
        self.verbose = verbose
        # Load hoi_list
        hoi_list_json = os.path.join(self.gt_dir,'hoi_list.json')
        self.hoi_list = io.load_json_object(hoi_list_json)
        
        ## replace str key with integer key
#        for i in range(len(self.hoi_list)):
#            self.hoi_list[i]['id'] = int(self.hoi_list[i]['id'])
        
        # Load subset ids to eval on
        self.img_names = self.dataset.samples
    
        # Create gt_dets
        print('Creating GT dets ...')
        self.gt_dets = load_gt_dets(self.gt_dir,self.img_names)
        self.pred_dets = None
        
        
    def evaluate_rand(self,model,device):
        print('localization type {}'.format('random'))
        ## inference on test set ##
        all_preds = []
        all_labels =[]
        all_positions_grid_a = []
        all_positions_grid_o = []
        all_cross_positions_grid_a = []
        all_cross_positions_grid_o = []
        
#        all_As_a = []
#        all_As_o = []
        
        all_img_names = []
        all_sizes = []
        
#        spatial_code = model.spatial_code.cpu().numpy()
        
        for i_batch, item in enumerate(self.dataloader):
            list_file_name,arr_feature_map,arr_label,arr_size = item
            with torch.no_grad():
                model.eval()
                features = arr_feature_map.to(device)
                
                out_package=model(features)
                preds = out_package['s'].cpu().numpy()
                positions_a = out_package['positions_a'].cpu().numpy() #[bk2]
                positions_o = out_package['positions_o'].cpu().numpy()
                
                labels = arr_label.cpu().numpy()
                arr_size = arr_size.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
                all_positions_grid_a.append(np.random.uniform(low=0.0, high=model.grid_size, size=positions_a.shape)) #all_positions_grid_a.append(max_positions_a)#
                all_positions_grid_o.append(np.random.uniform(low=0.0, high=model.grid_size, size=positions_o.shape)) #all_positions_grid_o.append(max_positions_o)#
                
#                all_As_a.append(A_a)
#                all_As_o.append(A_o)
                
                all_img_names.extend(list_file_name)
                all_sizes.append(arr_size)
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_positions_grid_a = np.concatenate(all_positions_grid_a)
        all_positions_grid_o = np.concatenate(all_positions_grid_o)
        
        
        all_sizes = np.concatenate(all_sizes)
        ## inference on test set ##
        type = 'single'
        
        
        ## package prediction
        label_preds = np.argsort(-all_preds,axis = 1)
        pred_dets = {}
        for i in range(all_preds.shape[0]):
            
            if self.k == -1:
                top_labels = label_preds[i]
            else:
                top_labels = label_preds[i][:self.k]
            
            pred_det = {}
            for label in top_labels:
                ## convert from grid coor to img coor ##
                pos_img_o = attention_2_location(coor_grid = all_positions_grid_o[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                pos_img_a = attention_2_location(coor_grid = all_positions_grid_a[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                ## convert from grid coor to img coor ##
                label_key = str(label+1).zfill(2)
                pred_det[label_key] = {'score':all_preds[i][label],
                                        'pos_grid_a':all_positions_grid_a[i][label],'pos_grid_o':all_positions_grid_o[i][label],
                                        'pos_img_a':pos_img_a[0],'pos_img_o':pos_img_o[0],
                                        'org_img_shape':all_sizes[i]} #'A_a':all_As_a[i][label],'A_o':all_As_o[i][label],
                
                if type in ['or','cross','or_random']:
                    cross_pos_img_o = attention_2_location(coor_grid = all_cross_positions_grid_o[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                    cross_pos_img_a = attention_2_location(coor_grid = all_cross_positions_grid_a[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                    pred_det[label_key]['cross_pos_img_a'] = cross_pos_img_a[0]
                    pred_det[label_key]['cross_pos_img_o'] = cross_pos_img_o[0]
                    
            
            pred_dets[all_img_names[i]] = pred_det
        ## package prediction
        
        ## store prediction ##
        self.pred_dets = pred_dets
        ## store prediction ##
        
        eval_inputs = []
        for hoi in self.hoi_list:
            eval_inputs.append(
                (hoi['id'],all_img_names,self.gt_dets,pred_dets,self.verbose,type))
    
        print(f'Starting a pool of {self.num_processes} workers ...')
        p = Pool(self.num_processes)
    
        print(f'Begin mAP computation ...')
        #output = []
        output = p.starmap(eval_hoi,eval_inputs)
        #output = eval_hoi(hoi['id'],all_img_names,self.gt_dets,pred_dets)

        p.close()
        p.join()

        AP_hoi = np.zeros(len(self.hoi_list))
        AP_human = np.zeros(len(self.hoi_list))
        AP_object = np.zeros(len(self.hoi_list))
        
        
        for ap_hoi,ap_human,ap_object,hoi_id in output:
            hoi_id_n = int(hoi_id)-1
            AP_hoi[hoi_id_n] = ap_hoi
            AP_human[hoi_id_n] = ap_human
            AP_object[hoi_id_n] = ap_object
        
        return (AP_hoi,AP_human,AP_object)
    
    def evaluate(self,model,device,type='single', metric = 'Top_Loc'):
        print('localization type {} metric {}'.format(type,metric))
        ## inference on test set ##
        all_preds = []
        all_labels =[]
        all_positions_grid_a = []
        all_positions_grid_o = []
        all_cross_positions_grid_a = []
        all_cross_positions_grid_o = []
        
#        all_As_a = []
#        all_As_o = []
        
        all_img_names = []
        all_sizes = []
        
#        spatial_code = model.spatial_code.cpu().numpy()
        
        for i_batch, item in enumerate(self.dataloader):
            list_file_name,arr_feature_map,arr_label,arr_size = item
            with torch.no_grad():
                model.eval()
                features = arr_feature_map.to(device)
                
                out_package=model(features)
                preds = out_package['s'].cpu().numpy()
                positions_a = out_package['positions_a'].cpu().numpy() #[bk2]
                positions_o = out_package['positions_o'].cpu().numpy()
                
#                A_a = out_package['A_a'].cpu().numpy() #[bkr]
#                A_o = out_package['A_o'].cpu().numpy()
#                
#                max_positions_a = spatial_code[np.argmax(A_a,axis = -1)]
#                max_positions_o = spatial_code[np.argmax(A_o,axis = -1)]
                
                labels = arr_label.cpu().numpy()
                arr_size = arr_size.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels)
                all_positions_grid_a.append(positions_a) #all_positions_grid_a.append(max_positions_a)#
                all_positions_grid_o.append(positions_o) #all_positions_grid_o.append(max_positions_o)#
                
#                all_As_a.append(A_a)
#                all_As_o.append(A_o)
                
                all_img_names.extend(list_file_name)
                all_sizes.append(arr_size)
                
                if type in ['or','cross']:
                    cross_positions_a = out_package['means_a'].cpu().numpy() #[bk2]
                    cross_positions_o = out_package['means_o'].cpu().numpy()
                    all_cross_positions_grid_a.append(cross_positions_a) 
                    all_cross_positions_grid_o.append(cross_positions_o)
                elif type == 'or_random':
                    cross_positions_a = np.random.uniform(low=0.0, high=model.grid_size, size=positions_a.shape) #[bk2]
                    cross_positions_o = np.random.uniform(low=0.0, high=model.grid_size, size=positions_o.shape)
                    all_cross_positions_grid_a.append(cross_positions_a) 
                    all_cross_positions_grid_o.append(cross_positions_o)
                
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_positions_grid_a = np.concatenate(all_positions_grid_a)
        all_positions_grid_o = np.concatenate(all_positions_grid_o)
        
#        all_As_a = np.concatenate(all_As_a)
#        all_As_o = np.concatenate(all_As_o)
        
        all_sizes = np.concatenate(all_sizes)
        
        if type in ['or','cross','or_random']:
            all_cross_positions_grid_a = np.concatenate(all_cross_positions_grid_a)
            all_cross_positions_grid_o = np.concatenate(all_cross_positions_grid_o)
        ## inference on test set ##
        
        ## package prediction
        label_preds = np.argsort(-all_preds,axis = 1)
        pred_dets = {}
        for i in range(all_preds.shape[0]):
            
            if self.k == -1:
                top_labels = label_preds[i]
            else:
                top_labels = label_preds[i][:self.k]
            
            pred_det = {}
            for label in top_labels:
                ## convert from grid coor to img coor ##
                pos_img_o = attention_2_location(coor_grid = all_positions_grid_o[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                pos_img_a = attention_2_location(coor_grid = all_positions_grid_a[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                ## convert from grid coor to img coor ##
                label_key = str(label+1).zfill(2)
                pred_det[label_key] = {'score':all_preds[i][label],
                                        'pos_grid_a':all_positions_grid_a[i][label],'pos_grid_o':all_positions_grid_o[i][label],
                                        'pos_img_a':pos_img_a[0],'pos_img_o':pos_img_o[0],
                                        'org_img_shape':all_sizes[i]} #'A_a':all_As_a[i][label],'A_o':all_As_o[i][label],
                
                if type in ['or','cross','or_random']:
                    cross_pos_img_o = attention_2_location(coor_grid = all_cross_positions_grid_o[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                    cross_pos_img_a = attention_2_location(coor_grid = all_cross_positions_grid_a[i][label][None], grid_size = model.grid_size, org_img_shape=all_sizes[i])
                    pred_det[label_key]['cross_pos_img_a'] = cross_pos_img_a[0]
                    pred_det[label_key]['cross_pos_img_o'] = cross_pos_img_o[0]
                    
            
            pred_dets[all_img_names[i]] = pred_det
        ## package prediction
        
        ## store prediction ##
        self.pred_dets = pred_dets
        ## store prediction ##
        
        eval_inputs = []
        for hoi in self.hoi_list:
            if metric == "GT_Known_Loc":
                skip_absent = True
            else:
                skip_absent = False
            eval_inputs.append(
                (hoi['id'],all_img_names,self.gt_dets,pred_dets,self.verbose,type,skip_absent))
    
        print(f'Starting a pool of {self.num_processes} workers ...')
        p = Pool(self.num_processes)
    
        print(f'Begin mAP computation ...')
        #output = []
        
        output = p.starmap(eval_hoi,eval_inputs)
        #output = eval_hoi(hoi['id'],all_img_names,self.gt_dets,pred_dets)

        p.close()
        p.join()

        AP_hoi = np.zeros(len(self.hoi_list))
        AP_human = np.zeros(len(self.hoi_list))
        AP_object = np.zeros(len(self.hoi_list))
        
        
        for ap_hoi,ap_human,ap_object,hoi_id in output:
            hoi_id_n = int(hoi_id)-1
            AP_hoi[hoi_id_n] = ap_hoi
            AP_human[hoi_id_n] = ap_human
            AP_object[hoi_id_n] = ap_object
        
        return (AP_hoi,AP_human,AP_object)