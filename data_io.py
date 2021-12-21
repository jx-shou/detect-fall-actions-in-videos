# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:20:00 2018

@author: zhou
"""




import torch as t
import pickle as pkl
import numpy as np



class FallDataset:
    def __init__(self, f_name):
        self.dataset = None
        with open(f_name, 'rb') as f:
            self.dataset = pkl.load(f)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        label = self.dataset[idx][1]
        
        data = data.astype('float32')
        data = np.multiply(data, 1/255.0)
        data = data.reshape(3,16,96,96)
        data = t.from_numpy(data)
        
        sample = {'video_data':data, 'video_label':label}
        return sample













