
import torch as t
from torch import nn



class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)        
        
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * 1, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(planes * 1),
                )
        

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample: residual = self.downsample(x)
            
        out += residual
    
        return out


class ResNet18_Enc(nn.Module):
    def __init__(self):
        super(ResNet18_Enc, self).__init__()
        
        self.stem = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), 
                          padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
                )

        self.middle = nn.Sequential(
                BasicBlock(64,64,1,False),
                BasicBlock(64,64,1,False),

                BasicBlock(64,128,2,True),
                BasicBlock(128,128,1,False),

                BasicBlock(128,256,2,True),
                BasicBlock(256,256,1,False),

                BasicBlock(256,512,2,True),
                BasicBlock(512,512,1,False),
                )

        self.avgpool = nn.AvgPool3d((1, 2, 2), stride=1)
  
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self,x):
        x = self.stem(x)
        x = self.middle(x)        
        x = self.avgpool(x)   #torch.Size([64, 512, 2, 2, 2])
        return x

class BasicBlockDec(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(BasicBlockDec, self).__init__()
        
        if upsample:
            stride = 2
            output_pad = 1
        else:
            stride = 1
            output_pad = 0

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.ConvTranspose3d(inplanes, planes, kernel_size=3, 
                                        stride=stride, padding=1, 
                                        output_padding=output_pad, bias=False)
        
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = nn.ConvTranspose3d(planes, planes, kernel_size=3, 
                                       stride=1, padding=1, bias=False)        

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.ConvTranspose3d(inplanes, planes, kernel_size=1, 
                                   stride=stride, output_padding=output_pad, 
                                   bias=False),
                nn.BatchNorm3d(planes)   
                )
                
    def forward(self, x):
        residual = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample: residual = self.downsample(x)
            
        out += residual
    
        return out  

    

class ResNet18_Dec(nn.Module):
    def __init__(self):
        super(ResNet18_Dec, self).__init__()
        self.head = nn.ConvTranspose3d(512,512,kernel_size=(1,4,4),stride=(1,2,2),padding=0)
        self.middle = nn.Sequential(
                BasicBlockDec(512,512,False),
                BasicBlockDec(512,256,True),
                
                BasicBlockDec(256,256,False),
                BasicBlockDec(256,128,True),
                
                BasicBlockDec(128,128,False),
                BasicBlockDec(128,64,True),
                
                BasicBlockDec(64,64,False),
                BasicBlockDec(64,3,True),
                )
        
  
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self,x):
        x = self.head(x)
        x = self.middle(x) 

        return x


class CVAE(nn.Module):
    def __init__(self, z_dim=500):
        super(CVAE, self).__init__()
        self.encoder = ResNet18_Enc()
        
        self.fc_mu = nn.Linear(2048, z_dim)
        self.fc_logvar = nn.Linear(2048, z_dim)
        self.fc3 = nn.Linear(z_dim, 2048)
        
        self.decoder = ResNet18_Dec()
        
    def reparameterize(self, mu, logvar):
        #logvar = log(sigma^2)
        std = logvar.mul(0.5).exp_()
        
        esp = t.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z



    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)   #torch.Size([64, 4096])
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.fc3(z)
        z = z.view(z.size(0),512,1,2,2)
        z = self.decoder(z)
    
        return z, mu, logvar







