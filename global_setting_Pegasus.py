#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:27:00 2019

@author: war-machince
"""

import torch
import numpy as np

seed = 214
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed) 

docker_path = './'
NFS_path = './'
