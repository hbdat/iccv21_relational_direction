# Interaction Compass: Multi-Label Zero-Shot Learning of Human-Object Interactions via Spatial Relations

## Overview
This repository contains the implementation of [Multi-Label Zero-Shot Learning of Human-Object Interactions via Spatial Relations](https://hbdat.github.io/pubs/iccv21_relation_direction_final.pdf).
> In this work, we develope a compositional model to recognize unseen human interactions based on spatial relations between human and objects.

![Image](https://github.com/hbdat/iccv21_relational_direction/raw/main/fig/schemantic_figure.png)

---
## Prerequisites
To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

---
## Data Preparation
1) Please download and extract information into the `./data` folder. We include details about download links as well as what are they used for in each folder within `./data' folder.

3) Please run feature extraction scripts in `./extract_feature` folder to extract features from the last convolution layers of ResNet as region features for the attention mechanism:
```
python ./extract_feature/hico_det/hico_det_extract_feature_map_ResNet_152_padding.py				                                    #create ./data/DeepFashion/feature_map_ResNet_101_DeepFashion_sep_seen_samples.hdf5
python ./extract_feature/visual_genome/visual_genome_extract_feature_map_ResNet_152_padding.py						            #create ./data/AWA2/feature_map_ResNet_101_AWA2.hdf5
```
as well as word embedding for zero-shot learning:
```
python ./extract_feature/hico_det/hico_extract_action_object_w2v.py						                                                                  #create ./data/CUB/feature_map_ResNet_101_CUB.hdf5
python ./extract_feature/visual_genome/visual_genome_extract_action_object_w2v.py						                                       #create ./data/SUN/feature_map_ResNet_101_SUN.hdf5
```

---
## Experiments
1) To train cross attention on HICO/VG datasets under different training splits (1A/1A2B), please run the following commands:
```
# HICO experiments
python ./experiments/hico_det/hico_det_pad_CrossAttention.py --partition train_1A --idx_GPU 1 --save_folder ./results/HICO_1A --mll_k_3 7 --mll_k_5 10 --loc_k 10             #1A setting
python ./experiments/hico_det/hico_det_pad_CrossAttention.py --partition train_1A2B --idx_GPU 2 --save_folder ./results/HICO_1A2B --mll_k_3 7 --mll_k_5 10 --loc_k 10     #1A2B setting

# Visual Genome experiments
python ./experiments/visual_genome_pad/VG_pad_CrossAttention.py --partition train_1A --idx_GPU 4 --save_folder ./results/VG_1A --mll_k_3 7 --mll_k_5 10               #1A setting
python ./experiments/visual_genome_pad/VG_pad_CrossAttention.py --partition train_1A2B --idx_GPU 5 --save_folder ./results/VG_1A2B --mll_k_3 7 --mll_k_5 10       #1A2B setting
```

---
## Pretrained Models
For ease of reproducing the results, we provided the pretrained models for:
Dataset | Setting | Model
--- |:---:|:---
HICO | 1A2B | [download](https://drive.google.com/file/d/1g8I_-WJVFpwZPeaf9qkjT_9i7p8s3xI8/view?usp=sharing)
HICO | 1A | [download](https://drive.google.com/file/d/1Jttz9iFNKT76ZOHnP6gswjOPDGwxSbqj/view?usp=sharing)
VisualGenome | 1A2B | [download](https://drive.google.com/file/d/1YO8HVcnXTDU7asY_cxX7OiMjp3v-UPRp/view?usp=sharing)
VisualGenome | 1A | [download](https://drive.google.com/file/d/1vEOFAsGNkcOAg7gKVTnAOxyXozpYRXWe/view?usp=sharing)

---
## Citation
If you find the project helpful, we would appreciate if you cite the works:
```
@article{Huynh:ICCV21,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Interaction Compass: Multi-Label Zero-Shot Learning of Human-Object Interactions via Spatial Relations},
  journal = {International Conference on Computer Vision},
  year = {2021}}
```

