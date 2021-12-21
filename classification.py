
import torch as t
from torch import nn
from torch.utils.data import DataLoader

from time import time

from data_io import FallDataset
from network import ResNet18






batch_size = 32
train_set = FallDataset('./dir/train_dataset')
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = FallDataset('./dir/test_dataset')
val_iter = DataLoader(val_set, batch_size=batch_size, shuffle=True)

net = ResNet18().cuda()
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

epoch = 100
for ep in range(epoch):
    tic = time()
    train_loss = 0.0
    train_acc = 0.0
    right_cnt = 0
    
    net.train()
    for i, batch in enumerate(train_iter):
        data = batch['video_data'].cuda()
        label = batch['video_label'].long().cuda()
        
        output = net(data)
        
        loss = cross_entropy_loss(output, label)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predict = t.max(output, 1)
        right_cnt += (predict == label).sum().item()
    # for

    train_loss /= len(train_set)
    train_acc = right_cnt / len(train_set)


    val_acc = 0.0
    right_cnt = 0
    
    net.eval()
    with t.no_grad():
        for val_batch in val_iter:
            val_data = val_batch['video_data'].cuda()
            val_label = val_batch['video_label'].long().cuda()
            
            val_output = net(val_data)
            _, predicted = t.max(val_output, 1)
            right_cnt += (predicted == val_label).sum().item()
        # for
    # with


    val_acc = right_cnt / len(val_set)
    tim = time() - tic
    
    print("[%2d/%2d] Loss:%.5f,Acc:%.3f,Val-Acc:%.3f,Time:%.1f" 
          %(ep, epoch, train_loss, train_acc, val_acc, tim))   

    t.save(net.state_dict(), './res18_black_hq225-240-he57-60_100ep.param')









