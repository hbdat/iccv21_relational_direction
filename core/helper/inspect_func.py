# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:49:14 2020

@author: badat
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F   
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
#%%
def compute_model_norms(model):
    dic_param_norms = {}
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            dic_param_norms[name] = torch.norm(param)
    
    return dic_param_norms