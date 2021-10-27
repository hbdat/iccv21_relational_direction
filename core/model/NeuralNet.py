# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:38:06 2020

@author: badat
"""


import torch
from torch import nn
import pdb
from torch.nn import functional as F

class NeuralNet(nn.Module):
    """ linear - relu - linear """

    def __init__(self, D_in, D_hidden, D_out , bias = True):
        """
        Args:
            D_in (int): Input dimension
            D_out (int): Output dimension
        """
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(in_features = D_in,
                                 out_features = D_hidden,
                                 bias=bias)     #the last dim must be the feature dim
        
        
        self.linear2 = nn.Linear(in_features = D_hidden,
                                 out_features = D_out,
                                 bias=bias)
        
        print('RELU')
        
    def forward(self, x):       #brf
        """
        """
        h = torch.relu(self.linear1(x))
        output = self.linear2(h)
            
        return output #
    
class ResNet(nn.Module):
    """ linear - relu - linear """

    def __init__(self, D_in, D_hidden, D_out , bias = True):
        """
        Args:
            D_in (int): Input dimension
            D_out (int): Output dimension
        """
        super(ResNet, self).__init__()
        self.linear1 = nn.Linear(in_features = D_in,
                                 out_features = D_hidden,
                                 bias=bias)     #the last dim must be the feature dim
        
        
        self.linear2 = nn.Linear(in_features = D_hidden,
                                 out_features = D_out,
                                 bias=bias)
        
        print('RELU')
        
    def forward(self, x):       #brf
        """
        """
        h = torch.relu(self.linear1(x))
        output = self.linear2(h) + x
            
        return output #bf,br,br
