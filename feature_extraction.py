# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:20:00 2018

@author: zhou
"""




import torch as t
from torch.utils.data import DataLoader

import pickle as pkl
import numpy as np

from data_io import FallDataset
from network import CVAE


batch_size = 32
dataset = FallDataset()
dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)


vae = CVAE().cuda()
file_path = './dir/file name'
vae.load_state_dict(t.load(file_path))
vae.eval()
print('load param.')



mu_list = []
lab_list = []
for i, batch in enumerate(dataloader):
    print(i)
    img = batch['video_data'].cuda()
    label = batch['video_label']
    
    recon_img, mu, logvar = vae(img)
    
    mu_detach = mu.detach().cpu().numpy()
    lab_detach = label.numpy()
    
    mu_list.append(mu_detach)
    lab_list.append(lab_detach)

    
mu_list = np.concatenate(mu_list, axis=0)
lab_list = np.concatenate(lab_list, axis=0)
print(mu_list.shape, lab_list.shape)

f_name = './dir/file_name'
with open(f_name, 'wb') as f:
    pkl.dump({'mu':mu_list, 'label':lab_list}, f)
    













