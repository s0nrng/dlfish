import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init

class SEBlock(nn.Module):
    def __init__(self, input_size, no_chans):
        super(SEBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(input_size),
            nn.Flatten(),
            nn.Linear(no_chans, 128),
            nn.ReLU(),
            nn.Linear(128, no_chans),
            nn.Unflatten(1, torch.Size([no_chans, 1])),
            nn.Sigmoid()
        )
        
    def forward(self, image):
        identity = image
        output = self.layers(image)
        identity = identity * output.unsqueeze(dim=2).expand_as(identity)
        return identity


class Bottleneck(nn.Module):
    def __init__(self, in_chans, k, mode = 'UP'):
        super(Bottleneck, self).__init__()
        self.mode = mode
        if mode == 'UP':
            self.bnorm1 = nn.BatchNorm2d(in_chans*k)
            self.conv1 = nn.Conv2d(in_chans*k, in_chans//2, 1, padding='same')
            init.xavier_uniform_(self.conv1.weight)
            init.constant_(self.conv1.bias, 0)
    
            self.bnorm2 = nn.BatchNorm2d(in_chans//2)
            self.conv2 = nn.Conv2d(in_chans//2, in_chans//2, 3, padding='same')
            init.xavier_uniform_(self.conv2.weight)
            init.constant_(self.conv2.bias, 0)
            
            self.bnorm3 = nn.BatchNorm2d(in_chans//2)
            self.conv3 = nn.Conv2d(in_chans//2, in_chans, 1, padding='same')
            init.xavier_uniform_(self.conv3.weight)
            init.constant_(self.conv3.bias, 0)
            
            self.bnorms = nn.BatchNorm2d(in_chans*k)
            self.shortcut = nn.Conv2d(in_chans*k, in_chans, 1, 1, bias=False)
            
        else:
            self.bnorm1 = nn.BatchNorm2d(in_chans)
            self.conv1 = nn.Conv2d(in_chans, in_chans//4, 1, padding='same')
            init.xavier_uniform_(self.conv1.weight)
            init.constant_(self.conv1.bias, 0)

            self.bnorm2 = nn.BatchNorm2d(in_chans//4)
            self.conv2 = nn.Conv2d(in_chans//4, in_chans//4, 3, padding='same')
            init.xavier_uniform_(self.conv2.weight)
            init.constant_(self.conv2.bias, 0)
            
            self.bnorm3 = nn.BatchNorm2d(in_chans//4)
            self.conv3 = nn.Conv2d(in_chans//4, in_chans, 1, padding='same')
            init.xavier_uniform_(self.conv3.weight)
            init.constant_(self.conv3.bias, 0)
        
    def forward(self, x):
        mode = self.mode
        identity = x
        x = self.conv1(f.relu(self.bnorm1(x)))
        x = self.conv2(f.relu(self.bnorm2(x)))
        x = self.conv3(f.relu(self.bnorm3(x)))
        
        if mode == 'UP':
            identity = self.shortcut(f.relu(self.bnorms(identity)))
        else:
            pass
        output = x + identity
        return output

class URBlock(nn.Module):
    def __init__(self, in_chans, k):
        super(URBlock, self).__init__()
        self.in_chans = in_chans
        self.k = k
        self.bottleneck = Bottleneck(in_chans, k)
        
    def forward(self, bs, ts):
        k = self.k
        Bs = torch.cat((bs, ts), dim=1)
        r = Bs.view(Bs.shape[0], Bs.shape[1]//k, k, Bs.shape[2], Bs.shape[3]).sum(2)
        M = self.bottleneck(Bs)
        output = M+r
        return output


class DRBlock(nn.Module):
    def __init__(self, in_chans, k):
        super(DRBlock, self).__init__()
        self.in_chans = in_chans
        self.bottleneck = Bottleneck(in_chans, k, mode='DOWN')
        
    def forward(self,tensor_list):
        Hs = torch.cat(tensor_list, dim=1)
        return Hs + self.bottleneck(Hs)


class FishTail(nn.Module):
    def __init__(self):
        super(FishTail, self).__init__()
        # 224x224 -> 112x112
        self.conv1 = nn.Conv2d(3, 16, 2, 2)
        init.xavier_uniform_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)
        self.bnorm1 = nn.BatchNorm2d(16)
        
        # 112x112 -> 56x56
        self.conv2 = nn.Conv2d(16, 64, 2, 2)
        init.xavier_uniform_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0)
        self.bnorm2 = nn.BatchNorm2d(64)
        
        # 56x56 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, 2, 2)
        init.xavier_uniform_(self.conv3.weight)
        init.constant_(self.conv3.bias, 0)
        self.bnorm3 = nn.BatchNorm2d(128)
        
        # 28x28 -> 14x14
        self.conv4 = nn.Conv2d(128, 256, 2, 2)
        init.xavier_uniform_(self.conv4.weight)
        init.constant_(self.conv4.bias, 0)
        self.bnorm4 = nn.BatchNorm2d(256)
        
        # 14x14 -> 7x7
        self.conv5 = nn.Conv2d(256, 512, 2, 2)
        init.xavier_uniform_(self.conv4.weight)
        init.constant_(self.conv4.bias, 0)
        self.bnorm5 = nn.BatchNorm2d(512)
        
        self.seblock = SEBlock(input_size=(7,7), no_chans=512)

    def forward(self, image):
        image = f.relu(self.conv1(image))
        
        t1 = self.bnorm2(self.conv2(image))
        
        t2 = f.relu(t1)
        t2 = self.bnorm3(self.conv3(t2))
        
        t3 = f.relu(t2)
        t3 = self.bnorm4(self.conv4(t3))
        
        t4 = f.relu(t3)
        t4 = self.bnorm5(self.conv5(t4))
        
        b4 = self.seblock(t4)
        return t1, t2, t3, t4, b4


class FishBody(nn.Module):
    def __init__(self):
        super(FishBody, self).__init__()
        self.up3 = nn.Sequential(
            nn.Conv2d(512, 256, 2, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.ur3 = URBlock(256, 2)
        self.bnorm3 = nn.BatchNorm2d(256)
        
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 2, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.ur2 = URBlock(128, 2)
        self.bnorm2 = nn.BatchNorm2d(128)
        
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 2, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.bnorm1 = nn.BatchNorm2d(64)
        self.ur1 = URBlock(64, 2)
        
    def forward(self, t1, t2, t3, t4, b4):
        b3 = f.interpolate(self.up3(b4), scale_factor=2)
        b3 = self.bnorm3(self.ur3(b3, t3))
        
        b2 = f.interpolate(self.up2(b3), scale_factor=2)
        b2 = self.bnorm2(self.ur2(b2, t2))
        
        b1 = f.interpolate(self.up1(b2), scale_factor=2)
        b1 = self.bnorm1(self.ur1(b1, t1))
        return b1, b2, b3


class FishHead(nn.Module):
    def __init__(self):
        super(FishHead, self).__init__()
        self.maxpool2 = nn.MaxPool2d(2)
        in_chans = 64 + 128*2
        self.dr2 = DRBlock(in_chans, k=None)
        self.bnorm2 = nn.BatchNorm2d(in_chans)
        self.maxpool3 = nn.MaxPool2d(2)
        in_chans += 256*2
        self.dr3 = DRBlock(in_chans, k=None)
        self.bnorm3 = nn.BatchNorm2d(in_chans)
        self.maxpool4 = nn.MaxPool2d(2)
        in_chans += 512*2
        self.dr4 = DRBlock(in_chans, None)
        self.bnorm4 = nn.BatchNorm2d(in_chans)
    def forward(self, t1, t2, t3, t4, b1, b2, b3, b4):
        h2 = self.maxpool2(b1)
        h2 = self.bnorm2(self.dr2((h2, b2, t2)))
        h3 = self.maxpool3(h2)
        h3 = self.bnorm3(self.dr3((h3, b3, t3)))
        h4 = self.maxpool4(h3)
        h4 = self.bnorm4(self.dr4((h4, b4, t4)))
        return h4

class FishNet(nn.Module):
    def __init__(self):
        super(FishNet, self).__init__()
        self.tail = FishTail()
        self.body = FishBody()
        self.head = FishHead()
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(1856),
            nn.ReLU(),
            nn.Conv2d(1856, 928, 1, bias=True),
            nn.BatchNorm2d(928),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(928, 500, 1, bias=True),
            nn.Flatten(),
            nn.Linear(500, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        t1, t2, t3, t4, b4 = self.tail(x)
        b1, b2, b3 = self.body(t1, t2, t3, t4, b4)
        output = self.head(t1, t2, t3, t4, b1, b2, b3, b4)
        output = self.classifier(output)
        return output