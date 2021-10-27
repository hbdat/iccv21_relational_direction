# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:43:05 2019

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
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
#%%
print('Loading pretrain w2v model')
model_name = 'word2vec-google-news-300'#best model
model = api.load(model_name)
dim_w2v = 300
print('Done loading model')
#%%
replace_word = [('','')]
#%%
path = './data/hico_20150920/hico_list_hoi.csv'
df = pd.read_csv(path,header = None, names = ['idx','obj','act'])
objects_unique = df['obj'].unique()
actions_unique = df['act'].unique()

object_idxs = [np.where(objects_unique == obj)[0][0] for obj in df['obj'].values]
action_idxs = [np.where(actions_unique == act)[0][0] for act in df['act'].values]
df["obj_idxs"] = object_idxs
df["act_idxs"] = action_idxs

Z_o = np.eye(len(objects_unique))[object_idxs]
Z_a = np.eye(len(actions_unique))[action_idxs]
#%%
df.to_csv("./data/hico_20150920/aug_hico_list_hoi.csv",index=None)
#%% pre-processing
def preprocessing(words):
    new_words = [' '.join(i.split('_')) for i in words]
    return new_words
#%%
print(">>>>actions<<<<")
actions_w2v = []
for s in preprocessing(actions_unique):
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    actions_w2v.append(w2v[np.newaxis,:])
actions_w2v=np.concatenate(actions_w2v,axis=0)
#%%
print(">>>>objects<<<<")
objects_w2v = []
for s in preprocessing(objects_unique):
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
    objects_w2v.append(w2v[np.newaxis,:])
objects_w2v=np.concatenate(objects_w2v,axis=0)
#%%
content = {'actions_w2v':actions_w2v,'objects_w2v':objects_w2v,'Z_a':Z_a,'Z_o':Z_o}
with open('./w2v/hico_act_obj.pkl','wb') as f:
    pickle.dump(content,f)