
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
from time import time
import numpy as np

from data_io import FallDataset
from network import CVAE

if not os.path.exists('./conv_vae_img'):
    os.mkdir('./conv_vae_img')




batch_size = 32
dataset = FallDataset('./dataset_dir/dataset_name')
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)


vae = CVAE().cuda()
print('CVAE')

criterion = nn.MSELoss(size_average=False)
optimizer = t.optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 510
for epoch in range(epochs):
    tt = time()
    train_loss = 0
    
    # random index to save reconstructed images during training
    aa = np.random.randint(len(dataset) // batch_size)
    bb = np.random.randint(batch_size)
    
    vae.train()
    for i, batch in enumerate(dataloader):
        img = batch['video_data'].cuda()
        recon_img, mu, logvar = vae(img)
        
        MSE = criterion(recon_img, img)
        KLD = -0.5 * t.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        optimizer.zero_grad()
        loss = MSE + KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if i == aa:
            saved_img = recon_img[bb].detach().cpu().data
            saved_img = saved_img.reshape(16,3,96,96)
            save_image(saved_img, './conv_vae_img/image_{}.png'.format(epoch))
    #for i, batch
    
    train_loss /= len(dataset)
    tim = time() - tt
    
    print("[%2d/%2d] loss:%5f, time:%.1f" %(epoch, epochs, train_loss, tim))
    
    if epoch % 100 == 0:
        t.save(vae.state_dict(), './vae-res18_' + str(epoch) + 'ep.param')

    







