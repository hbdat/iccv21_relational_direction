# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:42:13 2020

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
from core.helper.preprocessing_func import get_padding
#%% Coordinate conversion
def attention_2_bboxes(att,thres):
    '''
    Difficult to implement needed to take care of the multi-blobs situation
    '''
    pass

def attention_2_location(coor_grid, grid_size, org_img_shape): #coor [k2]
    coor = np.copy(coor_grid)
    assert len(coor.shape) == 2
    ## reverse height coordinate
#    coor[:,1] = (grid_size-1) - coor[:,1]
    
#    coor += 1
    
    max_size = np.max(org_img_shape)
    
    ## upscale
    upscale = max_size//grid_size
    coor *= upscale
    
    ## to original image
    l_pad, _, _, b_pad = get_padding(org_img_shape)
    coor[:,0] -= l_pad
    coor[:,1] -= b_pad
    
    return coor
    
def grid_coor_2_pad_coor(pred_coor,grid_size,test_input_size):
    ret = np.copy(pred_coor)
    upscale_ratio = test_input_size/grid_size
    ret *= upscale_ratio
    return ret

def pad_coor_2_img_coor(pred_coor, test_input_size, org_img_shape):
    ret = np.copy(pred_coor)
    org_h, org_w = org_img_shape 
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h) 
    dw = (test_input_size - resize_ratio * org_w) // 2 
    dh = (test_input_size - resize_ratio * org_h) // 2 
    ret[:, 0::2] = 1.0 * (ret[:, 0::2] - dw) / resize_ratio 
    ret[:, 1::2] = 1.0 * (ret[:, 1::2] - dh) / resize_ratio 
    return ret
#%% Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def custom_visualize_bboxes(img,bboxes,save_path):
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        
        # Create plot
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img[:,:,::-1])

        # Draw bounding boxes and labels of detections
        
        ### fix color ###
        bbox_colors = [colors[0]]*bboxes.shape[0] #random.sample(colors, n_cls_preds)
        ### fix color ###
        
        for idx in range(bboxes.shape[0]):

            w,h = img.shape[:2]
#            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            x1 = bboxes[idx][0]
            y1 = bboxes[idx][1]
            x2 = bboxes[idx][2]
            y2 = bboxes[idx][3]
            
#            assert x1 <= w and x2 <= w
#            assert y1 <= h and y2 <= h
            
            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[0]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s='None',
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
    #    filename = 'indx_{}'.format(idx)#path.split("/")[-1].split(".")[0]
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
#%% evaluation
def iou_xyxy_numpy(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU        

def matching_bboxes(gt_bboxes,pred_bboxes, iou_threshold=0.5):
    """
    """
#    n_match = 0
    match_pred_bboxes = []
    for gt_b in gt_bboxes:
        for idx_b,pred_b in enumerate(pred_bboxes):
            iou = iou_xyxy_numpy(gt_b, pred_b)
            match = iou > iou_threshold
            if match:
#                n_match += 1
                match_pred_bboxes.append(idx_b)
                break
    return np.array(match_pred_bboxes)