# -*- coding: utf-8 -*-   
"""   
Created on Tue Aug  4 17:56:07 2020   
   
@author: badat   
"""   
       
import torch       
import torch.nn as nn       
import torch.nn.functional as F       
import numpy as np       
#%%       
from core.model.NeuralNet import NeuralNet, ResNet 
import pdb       
#%%       
class CrossAttention(nn.Module):       
    #####       
    # einstein sum notation       
    # b: Batch size \ f: dim feature \ v: dim w2v \ r: number of region \ k: number of classes       
    # i: number of attribute \ h : hidden attention dim       
    #####       
    def __init__(self,dim_f,dim_v,     
                 init_w2v_a,init_w2v_o,     
                 Z_a,Z_o,     
                 lamb=-1,  
                 trainable_w2v_a = True,trainable_w2v_o = True,      
                 normalize_V_a = True, normalize_V_o = True, normalize_F = True,     
                 label_type = 'interaction',grid_size=7, is_cross = True, 
                 is_w2v_map = False):       
        super(CrossAttention, self).__init__()       
             
        self.label_type = label_type     
        assert self.label_type in ['interaction','action','object','object_via_interact']     
             
        self.dim_f = dim_f       
        self.dim_v = dim_v       
        self.grid_size = grid_size   
        self.is_cross = is_cross   
        self.lamb = lamb  
         
        self.is_w2v_map = is_w2v_map 
         
        if self.lamb < 0: 
            trainable_weight_cross = True 
        else: 
            trainable_weight_cross = False 
         
        if is_w2v_map: 
            trainable_w2v_a = trainable_w2v_o = False 
            self.w2v_map = ResNet(D_in = self.dim_v,   
                                               D_hidden=self.dim_v//5,    
                                               D_out = self.dim_v)   
             
             
        self.init_w2v_a = F.normalize(torch.tensor(init_w2v_a)).float()     
        self.V_a = nn.Parameter(self.init_w2v_a.clone(),requires_grad = trainable_w2v_a)       
             
        self.init_w2v_o = F.normalize(torch.tensor(init_w2v_o)).float()     
        self.V_o = nn.Parameter(self.init_w2v_o.clone(),requires_grad = trainable_w2v_o)      
             
        self.W_a = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#       
        self.W_o = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#       
        self.W_e_a = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#       
        self.W_e_o = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#       
             
        self.Z_a = nn.Parameter(torch.tensor(Z_a).float(),requires_grad = False)      
        self.Z_o = nn.Parameter(torch.tensor(Z_o).float(),requires_grad = False)      
     
        self.normalize_V_a = normalize_V_a     
        self.normalize_V_o = normalize_V_o     
        self.normalize_F = normalize_F       
             
        self.log_softmax_func = nn.LogSoftmax(dim=1)     
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction = 'none')     
          
        if self.is_cross: 
            ### cross-attention ###   
            self.object2actor_gaussian = NeuralNet(D_in = self.dim_v+self.dim_f+2,   
                                                   D_hidden=300,    
                                                   D_out = 4)   
            self.actor2object_gaussian = NeuralNet(D_in = self.dim_v+self.dim_f+2,   
                                                   D_hidden=300,    
                                                   D_out = 4)   
           
        self.weight_cross_a = nn.Parameter(torch.zeros(1).float(),requires_grad = trainable_weight_cross)   
        self.weight_cross_o = nn.Parameter(torch.zeros(1).float(),requires_grad = trainable_weight_cross)   
             
        spatial_code = torch.zeros(self.grid_size,self.grid_size,2)   
        spatial_code[:,:,0] = torch.arange(self.grid_size)[None,:].repeat(self.grid_size,1)   
        spatial_code[:,:,1] = torch.arange(self.grid_size)[:,None].repeat(1,self.grid_size)   
        spatial_code = spatial_code.reshape(-1,2)   
           
        self.spatial_code = nn.Parameter(spatial_code,requires_grad = False)   
         
         
          
         
        ### cross-attention ###   
          
        print('-'*30)       
        print('Configuration')       
             
        print('lamb {}'.format(lamb))  
             
        if self.normalize_V_a:       
            print('normalize V_a')       
        else:       
            print('no constraint V_a')     
                 
        if self.normalize_V_o:       
            print('normalize V_o')       
        else:       
            print('no constraint V_o')     
                   
        if self.normalize_F:       
            print('normalize F')       
        else:       
            print('no constraint F')       
           
    def compute_V_a(self):     
        V_a = self.V_a 
        if self.is_w2v_map: 
            V_a = self.w2v_map(V_a) 
         
        if self.normalize_V_a:       
            V_a_n = F.normalize(V_a)     
        else:       
            V_a_n = V_a       
        return V_a_n     
         
    def compute_V_o(self):     
        V_o = self.V_o 
         
        if self.is_w2v_map: 
            V_o = self.w2v_map(V_o) 
         
        if self.normalize_V_o:       
            V_o_n = F.normalize(V_o)     
        else:       
            V_o_n = V_o       
        return V_o_n     
             
    def binary_cross_entropy(self,s,labels):

        labels = labels.float()      
              
        indicator = labels.clone()      
        indicator[indicator<1] = 0      
              
        loss = self.bce_criterion(s,indicator)      
              
        mask = loss.new_ones(loss.shape).to(loss.device)      
           
        mask = mask.masked_fill(labels == 0, 0)      
              
        loss = torch.einsum('bk,bk->b',mask,loss)      
        label_count = torch.einsum('bk->b',mask)      
              
        loss = torch.mean(loss/label_count)      
              
        return loss  
    
    def compute_loss(self,in_package):     
        ## total loss  
          
        labels = in_package['labels']     
        labels = labels.float()    
        
        loss = self.binary_cross_entropy(s=in_package['s'],labels=in_package['labels'])#loss_self + 0.0*loss_cross  
          
        out_package = {'loss':loss,'weight_cross_a':self.weight_cross_a,'weight_cross_o':self.weight_cross_o}       
               
        return out_package       
       
       
    def compute_gaussian_mask(self,means,std_variances):    #[bk2],[bk2]   
        Zs = self.spatial_code[None,None,:,:] - means[:,:,None,:]      #[b,k,r,2] <== [1,1,r,2] - [b,k,1,2]<== [r2] - [bk2]   
        Zs = Zs / std_variances[:,:,None,:]                             #[bkr2]/[b,k,1,2] <== [bkr2] - [bk2]   
        Zs = torch.pow(Zs,2)               #[bkr2]   
        Zs = torch.sum(Zs,dim=-1)           #[bkr]   
        Es = torch.exp(-Zs/2.0)                 #[bkr]   
        normalized_cooefs = 2*3.14*torch.prod(std_variances,dim = -1)  #[bk]<=[bk2]   
           
        Masks = Es/normalized_cooefs[:,:,None]      #[bkr]   
           
        #assert  torch.sum(torch.sum(Masks,dim=-1) - 1.0) < 1e-5   
           
        return Masks   
           
           
    def infer_gaussian_parameters(self,As,Hs,V_a,net):      #[bkr], [bkf], [kv]   
        positions = torch.einsum('bkr,rj->bkj',As,self.spatial_code)    #[bk2]   
        b = As.shape[0]   
        inp = torch.cat([Hs,positions,V_a[None].repeat(b,1,1)],dim=-1)         #[bkf,bkj,kv]   
        gaussian_params = net(inp)           #[bk4]   
           
        delta_means = torch.tanh(gaussian_params[:,:,:2]) *self.grid_size     #[bk2]               #[bk2]   
        std_variances = torch.sigmoid(gaussian_params[:,:,2:]) *self.grid_size #[bk2]             #[bk2]   
           
        means = positions+delta_means   
           
        return means,std_variances   
       
    def compute_gaussian_attention(self,As,Hs,V_a,net):   
        means,std_variances = self.infer_gaussian_parameters(As,Hs,V_a,net)   
        Masks = self.compute_gaussian_mask(means,std_variances)   
        return Masks, means, std_variances   
       
    def forward(self,Fs):            #Fs [brf] labels [bk] << need to implement mapping from index to w2v      
        shape = Fs.shape     
         
        if len(shape) == 4:   
            Fs = Fs.reshape(shape[0],shape[1],shape[2]*shape[3])     
            Fs = Fs.permute(0,2,1) #[brf] <== [bfr]   
               
        # please being careful with normalization     
        V_a_n = self.compute_V_a()     
        V_o_n = self.compute_V_o()     
            
        assert Fs.shape[-1] == self.dim_f    
            
        if self.normalize_F:       
            Fs = F.normalize(Fs,dim = -1)     
             
        A_o = torch.einsum('ov,vf,brf->bor',V_o_n,self.W_o,Fs)   
        A_o_log = A_o   
        A_o = F.softmax(A_o,dim = -1)     
        Hs_o = torch.einsum('bor,brf->bof',A_o,Fs)     
             
        A_a = torch.einsum('av,vf,brf->bar',V_a_n,self.W_a,Fs)    
        A_a_log = A_a   
        A_a = F.softmax(A_a,dim = -1)     
        Hs_a = torch.einsum('bar,brf->baf',A_a,Fs)     
             
        e_a = torch.einsum('av,vf,baf->ba',V_a_n,self.W_e_a,Hs_a)     
        e_o = torch.einsum('ov,vf,bof->bo',V_o_n,self.W_e_o,Hs_o)     
             
        s_a = torch.einsum('ba,ka->bk',e_a,self.Z_a)     
        s_o = torch.einsum('bo,ko->bk',e_o,self.Z_o)     
           
        ### replicate object/action according to possible interaction ###   
        A_o_up = torch.einsum('bor,ko->bkr',A_o,self.Z_o)   
        A_a_up = torch.einsum('bar,ka->bkr',A_a,self.Z_a)   
           
        A_o_log_up = torch.einsum('bof,ko->bkf',A_o_log,self.Z_o)   
        A_a_log_up = torch.einsum('baf,ka->bkf',A_a_log,self.Z_a)   
           
        V_a_up = torch.einsum('av,ka->kv',V_a_n,self.Z_a)   
        V_o_up = torch.einsum('ov,ko->kv',V_o_n,self.Z_o)   
           
        Hs_o_up = torch.einsum('bof,ko->bkf',Hs_o,self.Z_o)   
        Hs_a_up = torch.einsum('baf,ka->bkf',Hs_a,self.Z_a)   
        ### replicate object/action according to possible interaction ###   
           
        ### Cross-Attention ###   
        if self.is_cross: 
            Masks_a, means_a, std_variances_a = self.compute_gaussian_attention(As=A_o_up,Hs=Hs_o_up,V_a=V_a_up,net=self.object2actor_gaussian)  #[bkr]   
            Masks_o, means_o, std_variances_o = self.compute_gaussian_attention(As=A_a_up,Hs=Hs_a_up,V_a=V_a_up,net=self.actor2object_gaussian)  #[bkr]   
             
            Hs_o_cross = torch.einsum('bkr,brf->bkf',Masks_o,Fs)   
             
            Hs_a_cross = torch.einsum('bkr,brf->bkf',Masks_a,Fs)   
            s_a_cross = torch.einsum('kv,vf,bkf->bk',V_a_up,self.W_e_a,Hs_a_cross)   
            s_o_cross = torch.einsum('kv,vf,bkf->bk',V_o_up,self.W_e_o,Hs_o_cross) 
        ### Cross-Attention ###   
           
           
           
        if self.label_type == 'interaction':     
            if self.is_cross:  
                if self.lamb == -1: 
                    s = s_a + s_o + (s_a_cross*self.weight_cross_a + s_o_cross*self.weight_cross_o)#*self.lamb#  
                else: 
                    s = s_a + s_o + (s_a_cross + s_o_cross)*self.lamb 
                 
                s_self = s_a + s_o  
                s_cross = s_a_cross + s_o_cross  
            else:   
                s = s_a + s_o 
        elif self.label_type == 'action':     
            s = None     
        elif self.label_type == 'object':     
            s = None     
        elif self.label_type == 'object_via_interact':   
            if self.training:   
                s = None    
            else:   
                s = None    
         
        positions_o = torch.einsum('bkr,rj->bkj',A_o_up,self.spatial_code) 
        positions_a = torch.einsum('bkr,rj->bkj',A_a_up,self.spatial_code) 
         
        package = {'s':s, 'e_a':e_a,'e_o':e_o, 'A_o':A_o_up, 'A_a':A_a_up, 'A_o_log':A_o_log_up, 'A_a_log':A_a_log_up, 
                   'positions_o':positions_o,'positions_a':positions_a}  #[bk2]  
         
        if self.is_cross: 
            package_cross = {'Masks_o':Masks_o,'Masks_a':Masks_a,   
                           'means_a':means_a, 'std_variances_a':std_variances_a,   
                           'means_o':means_o, 'std_variances_o':std_variances_o,  
                           's_cross': s_cross,'s_self': s_self} 
             
            package = {**package,**package_cross} 
         
        return package     
