# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 23:09:25 2020

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
#%%
from core.dataset.HICODet_pad_Dataset import HICODet_pad_Dataset
from core.model.CrossAttention import CrossAttention
from core.helper.helper_func import get_bbox_features,evaluate_mAP,Logger,evaluate_k
from core.helper.localize_eval import LocEvaluator_HICODet
#%%
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--idx_GPU', type=int, default=0, help='')
parser.add_argument('--save_folder', type=str, default='', help='')
parser.add_argument('--load_model', type=str, default='', help='')
parser.add_argument('--comment', type=str, default='', help='')
parser.add_argument('--trainable_w2v', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--normalize_V', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--lamb', type=float, default=-1)
parser.add_argument('--is_w2v_map', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--loc_k', type=int, default=3, help='')
parser.add_argument('--mll_k_3', type=int, default=3)
parser.add_argument('--mll_k_5', type=int, default=5)
parser.add_argument('--partition', type=str, default='', help='')
opt = parser.parse_args() 
'''
python ./experiments/visual_genome_pad/1A/VG_pad_DAZLE_1A.py --idx_GPU 5 --save_folder 'trainable_w2v_no_normalize' --trainable_w2v True --normalize_V False
'''
#%%
batch_size = 32
epochs = 10
label_type = 'interaction'
idx_GPU = opt.idx_GPU
save_folder =  opt.save_folder

comment = opt.comment

is_save = True

print('-'*30)
print('label_type {}'.format(label_type))
print('-'*30)
#%%
with open('./w2v/hico_act_obj.pkl','rb') as f:
    content = pickle.load(f)
#%%
partition = opt.partition
train_hicoDataset = HICODet_pad_Dataset(partition,content,label_type)
test_hicoDataset = HICODet_pad_Dataset('test',content,label_type)

train_dataloader = torch.utils.data.DataLoader(train_hicoDataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test_hicoDataset,
                                             batch_size=batch_size*10, shuffle=False,
                                             num_workers=4)
k=3
evaluator = LocEvaluator_HICODet(dataloader=test_dataloader,num_processes=12,k=opt.loc_k)
#%%
model = CrossAttention(dim_f=2048,dim_v=300,
                 init_w2v_a=content['actions_w2v'],init_w2v_o=content['objects_w2v'],
                 Z_a=content['Z_a'],Z_o=content['Z_o'],
                 trainable_w2v_a = opt.trainable_w2v,trainable_w2v_o = opt.trainable_w2v, 
                 normalize_V_a = opt.normalize_V, normalize_V_o = opt.normalize_V, normalize_F = True,
                 label_type = label_type, grid_size=train_hicoDataset.grid_size,
                 lamb = opt.lamb, is_w2v_map = opt.is_w2v_map)

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
model.to(device)

if opt.load_model != '':
    model.load_state_dict(torch.load(opt.load_model,map_location=device))

#%%
params_to_update = []
params_names = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t",name)
#%%
lr = 0.001
weight_decay = 0.#0.0001
momentum = 0.#0.#
#%%
optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)
#%%
experiment_dir = NFS_path+'results/{}/LOC_HICODet_pad_CrossAttention_k_{}_{}_{}_GPU_{}_partition_{}_time_{}/'.format(save_folder,opt.loc_k,comment,label_type,idx_GPU,partition,str(time.time()).replace('.','d'))
if is_save:
    os.makedirs(experiment_dir)
    with open(experiment_dir+'config.txt','w') as f:
        f.writelines(str(opt))
logger=Logger(experiment_dir+'stats.csv',['loss','mAP_all', 'mAP_seen', 'mAP_unseen','mAP_interact',
                                          'f1_5_all', 'f1_5_seen', 'f1_5_unseen', 'f1_5_interact',
                                          'f1_3_all', 'f1_3_seen', 'f1_3_unseen', 'f1_3_interact',
                                          'weight_cross_a','weight_cross_o'])

logger_part_F1_3 = Logger(experiment_dir+'partition_F1_3.csv',['1A','2A','1B','2B']) 
logger_part_F1_5 = Logger(experiment_dir+'partition_F1_5.csv',['1A','2A','1B','2B'])      
logger_part_AP = Logger(experiment_dir+'partition_AP.csv',['1A','2A','1B','2B']) 
    
logger_detail_hoi = Logger(experiment_dir+'detail_hoi.csv',['mAP_all', 'mAP_seen', 'mAP_unseen','mAP_interact','1A','2A','1B','2B'])
logger_detail_human = Logger(experiment_dir+'detail_human.csv',['mAP_all', 'mAP_seen', 'mAP_unseen','mAP_interact','1A','2A','1B','2B'])
logger_detail_object = Logger(experiment_dir+'detail_object.csv',['mAP_all', 'mAP_seen', 'mAP_unseen','mAP_interact','1A','2A','1B','2B'])

logger_ranking_k_3 = Logger(experiment_dir+'ranking_k_{}.csv'.format(opt.mll_k_3),['f1_3_all','p_3_all','r_3_all','f1_3_seen','p_3_seen','r_3_seen',
                                                         'f1_3_unseen','p_3_unseen','r_3_unseen','f1_3_interact','p_3_interact','r_3_interact'])
    
logger_ranking_k_5 = Logger(experiment_dir+'ranking_k_{}.csv'.format(opt.mll_k_5),['f1_5_all','p_5_all','r_5_all','f1_5_seen','p_5_seen','r_5_seen',
                                                         'f1_5_unseen','p_5_unseen','r_5_unseen','f1_5_interact','p_5_interact','r_5_interact'])
#%%
if partition == 'train_1A2B':
    seen_idxs = np.concatenate([train_hicoDataset.partition_1A,train_hicoDataset.partition_2B])
    unseen_idxs = np.concatenate([train_hicoDataset.partition_2A,train_hicoDataset.partition_1B])
    interact_idxs = np.concatenate([seen_idxs,unseen_idxs])
elif partition == 'train_1A': 
    seen_idxs = np.concatenate([train_hicoDataset.partition_1A]) 
    unseen_idxs = np.concatenate([train_hicoDataset.partition_2A,train_hicoDataset.partition_1B,train_hicoDataset.partition_2B]) 
    interact_idxs = np.concatenate([seen_idxs,unseen_idxs]) 

def package_mAP(AP,logger_detail):
    mAP_all, mAP_seen, mAP_unseen, mAP_interact = np.mean(AP),np.mean(AP[seen_idxs]),np.mean(AP[unseen_idxs]),np.mean(AP[interact_idxs])
    logger_detail.add([mAP_all, mAP_seen, mAP_unseen, mAP_interact,
                       np.mean(AP[train_hicoDataset.partition_1A]),np.mean(AP[train_hicoDataset.partition_2A]), 
                       np.mean(AP[train_hicoDataset.partition_1B]),np.mean(AP[train_hicoDataset.partition_2B])])
    
    print(logger_detail.df.iloc[-1])
    if is_save:
        logger_detail.save()
    
for epoch in range(epochs):
#    start_time = time.time()
    for i_batch, (arr_file_name,arr_feature_map,arr_label) in enumerate(train_dataloader):
#        print(i_batch)
        arr_feature_map = arr_feature_map.to(device)
        
        features=arr_feature_map #[B,K,C] == [brf]
        
#        pdb.set_trace()
        
        features = features.to(device)
        labels = arr_label.to(device)
        
        
        model.train()
        optimizer.zero_grad()
        
        out_package=model(features)
        in_package = out_package
        in_package['labels'] = labels
        
        out_package = model.compute_loss(in_package)
        
        loss = out_package['loss']
        
        loss.backward()
        optimizer.step()
        
        if i_batch % 100 == 0:
            print(i_batch)
            print(out_package)
            AP,all_preds,all_labels=evaluate_mAP(test_dataloader,model,device)
            
            f1_3_all,p_3_all,r_3_all = evaluate_k(opt.mll_k_3, None, None, None, all_preds,all_labels) 
            f1_5_all,p_5_all,r_5_all = evaluate_k(opt.mll_k_5, None, None, None, all_preds,all_labels) 
             
            f1_3_seen,p_3_seen,r_3_seen = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,seen_idxs],all_labels[:,seen_idxs]) 
            f1_5_seen,p_5_seen,r_5_seen = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,seen_idxs],all_labels[:,seen_idxs]) 
             
            f1_3_unseen,p_3_unseen,r_3_unseen = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,unseen_idxs],all_labels[:,unseen_idxs]) 
            f1_5_unseen,p_5_unseen,r_5_unseen = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,unseen_idxs],all_labels[:,unseen_idxs]) 
             
            f1_3_interact,p_3_interact,r_3_interact = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,interact_idxs],all_labels[:,interact_idxs]) 
            f1_5_interact,p_5_interact,r_5_interact = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,interact_idxs],all_labels[:,interact_idxs]) 
            
            mAP_all, mAP_seen, mAP_unseen, mAP_interact = np.mean(AP),np.mean(AP[seen_idxs]),np.mean(AP[unseen_idxs]),np.mean(AP[interact_idxs])
            
            print("mAP_all: {} mAP_seen: {} mAP_unseen: {}, mAP_interact: {}".format(mAP_all, mAP_seen, mAP_unseen, mAP_interact))
            print("f1_5_all: {} f1_5_seen: {} f1_5_unseen: {} f1_5_interact {}".format(f1_5_all, f1_5_seen, f1_5_unseen, f1_5_interact))
            print("f1_3_all: {} f1_3_seen: {} f1_3_unseen: {} f1_3_interact {}".format(f1_3_all, f1_3_seen, f1_3_unseen, f1_3_interact))
            
            logger.add([loss.item(),mAP_all, mAP_seen, mAP_unseen,mAP_interact,
                        f1_5_all, f1_5_seen, f1_5_unseen, f1_5_interact,
                        f1_3_all, f1_3_seen, f1_3_unseen, f1_3_interact,
                        model.weight_cross_a.item(), model.weight_cross_o.item()])
    
            logger_ranking_k_3.add([f1_3_all,p_3_all,r_3_all,f1_3_seen,p_3_seen,r_3_seen,
                                    f1_3_unseen,p_3_unseen,r_3_unseen,f1_3_interact,p_3_interact,r_3_interact])
    
            logger_ranking_k_5.add([f1_5_all,p_5_all,r_5_all,f1_5_seen,p_5_seen,r_5_seen,
                                    f1_5_unseen,p_5_unseen,r_5_unseen,f1_5_interact,p_5_interact,r_5_interact])       
     
            full_1A = train_hicoDataset.partition_1A 
            full_2A = train_hicoDataset.partition_2A 
            full_1B = train_hicoDataset.partition_1B  
            full_2B = train_hicoDataset.partition_2B 
             
            f1_3_1A,_,_ = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,full_1A],all_labels[:,full_1A]) 
            f1_3_2A,_,_ = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,full_2A],all_labels[:,full_2A]) 
            f1_3_1B,_,_ = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,full_1B],all_labels[:,full_1B]) 
            f1_3_2B,_,_ = evaluate_k(opt.mll_k_3, None, None, None, all_preds[:,full_2B],all_labels[:,full_2B]) 
            logger_part_F1_3.add([f1_3_1A,f1_3_2A,f1_3_1B,f1_3_2B]) 
             
            f1_5_1A,_,_ = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,full_1A],all_labels[:,full_1A]) 
            f1_5_2A,_,_ = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,full_2A],all_labels[:,full_2A]) 
            f1_5_1B,_,_ = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,full_1B],all_labels[:,full_1B]) 
            f1_5_2B,_,_ = evaluate_k(opt.mll_k_5, None, None, None, all_preds[:,full_2B],all_labels[:,full_2B]) 
            logger_part_F1_5.add([f1_5_1A,f1_5_2A,f1_5_1B,f1_5_2B]) 
             
            logger_part_AP.add([np.mean(AP[full_1A]),np.mean(AP[full_2A]), 
                                np.mean(AP[full_1B]),np.mean(AP[full_2B])]) 
            
            logger_ranking_k_3.add([f1_3_all,p_3_all,r_3_all,f1_3_seen,p_3_seen,r_3_seen,
                                    f1_3_unseen,p_3_unseen,r_3_unseen,f1_3_interact,p_3_interact,r_3_interact])
    
            logger_ranking_k_5.add([f1_5_all,p_5_all,r_5_all,f1_5_seen,p_5_seen,r_5_seen,
                                    f1_5_unseen,p_5_unseen,r_5_unseen,f1_5_interact,p_5_interact,r_5_interact]) 
    
            if is_save:
                logger.save()
                logger_ranking_k_3.save()
                logger_ranking_k_5.save()
                logger_part_AP.save() 
                logger_part_F1_3.save()
                logger_part_F1_5.save()
        
        if i_batch % 500 == 0:
            AP_hoi,AP_human,AP_object = evaluator.evaluate(model,device)
            
            print('hoi')
            package_mAP(AP_hoi,logger_detail_hoi)
            print('human')
            package_mAP(AP_human,logger_detail_human)
            print('object')
            package_mAP(AP_object,logger_detail_object)
            
            print("mAP_hoi {} mAP_human {} mAP_object {}".format(np.mean(AP_hoi),np.mean(AP_human),np.mean(AP_object)))

            
#%%
if is_save:
    torch.save(model.state_dict(), experiment_dir+'model_final.pt')
#%%
#if is_save:
#    if label_type == 'action':    
#        df = pd.DataFrame()
#        df['act'] = pd.read_csv('./data/hico_20150920/act_list.csv')['act']
#        df['AP'] = AP
#        df.to_csv(experiment_dir+'breakdown_act.csv')
#    elif label_type == 'interaction':
#        df = pd.read_csv('./data/hico_20150920/hico_list_hoi.csv',header = None)
#        df['AP'] = AP
#        df.to_csv(experiment_dir+'breakdown_interaction.csv')
