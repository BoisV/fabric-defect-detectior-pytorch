from torch import nn
from torch.utils.data.dataloader import DataLoader
from lib.utils.dataset import FabricDataset
import torch
import torchvision
from torchvision import transforms
from torch import cuda
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda:0' if cuda.is_available() else 'cpu')


class fabric(nn.Module):
    def __init__(self):
        super(fabric, self).__init__()
        x=torchvision.models.resnet18(num_classes=15)
        
        self.feature=nn.Sequential(
            x.conv1,
            x.bn1,
            x.relu,
            x.maxpool,
            x.layer1,
            x.layer2,
            x.layer3,
            x.layer4

        )
        self.avgpool=x.avgpool
        self.classifier=x.fc

    def forward(self, xx, xt):
        y = self.feature(xx)
        ytemp = self.feature(xt)
        y=self.avgpool(y)
        ytemp=self.avgpool(ytemp)
        c = self.classifier(torch.flatten(y-ytemp, 1))
        return c
def pretrain():
    
    train = FabricDataset(root='./data/X_fabric_data/tr/')
    test = FabricDataset(root='./data/X_fabric_data/te/')
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 20
    lr = 0.1

    model=fabric()
    
    model = model.to(device)
    if cuda.is_available():
        model = torch.nn.DataParallel(model)
    opt1 = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(MAX_EPOCHES):
        for xx, xt, Y in train:

            xx = xx.to(device)
            xt = xt.to(device)
            c = model(xx, xt)

            Y = Y.to(device)
            loss = criterion(c, Y)

            opt1.zero_grad()
            loss.backward()

            opt1.step()
            print(loss.item())

        with torch.no_grad():

            right = 0
            su = 0
            for xx, xt, Y in test: 
                xx = xx.to(device)
                xt = xt.to(device)
                c = model(  xx, xt)
                su = su+Y.shape[0]

                right = right+(c.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()

            print('test', epoch, right/su)
    if cuda.is_available():
        return model.module.feature
    else:
        return model.feature

def finetune(feature):

    xxx=torchvision.models.resnet18(num_classes=4)
        
    classifier=xxx.fc

    train = FabricDataset(root='./data/DIY_fabric_data/tr/')
    test = FabricDataset(root='./data/DIY_fabric_data/te/')
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 30
    lr = 0.1
 
    
    feature = feature.to(device)
    classifier=classifier.to(device)
    if cuda.is_available():
        feature = torch.nn.DataParallel(feature)
        classifier = torch.nn.DataParallel(classifier)
    
    opt1=torch.optim.Adam([
            {'params':feature.parameters(),'lr':0.0001},
            
            {'params':classifier.parameters(),'lr':0.01}
            ])
    criterion = nn.CrossEntropyLoss()
    avgpool=xxx.avgpool
    for epoch in range(MAX_EPOCHES):
        for xx, xt, Y in train:

            xx = xx.to(device)
            xt = xt.to(device)
            
            
            y = feature(xx)
            ytemp = feature(xt)
            y= avgpool(y)
            ytemp= avgpool(ytemp)
            c =  classifier(torch.flatten(y-ytemp, 1))

            Y[Y == 1] = 0
            Y[Y == 2] = 1
            Y[Y == 5] = 2
            Y[Y == 13] = 3
            Y = Y.to(device) 
            loss = criterion(c, Y)

            opt1.zero_grad()
            loss.backward()

            opt1.step()
            print(loss.item())

        with torch.no_grad():

            right = 0
            su = 0
            for xx, xt, Y in test: 
                xx = xx.to(device)
                xt = xt.to(device)
                
                
                y = feature(xx)
                ytemp = feature(xt)
                y= avgpool(y)
                ytemp= avgpool(ytemp)
                c =  classifier(torch.flatten(y-ytemp, 1))
                su = su+Y.shape[0]

                Y[Y == 1] = 0
                Y[Y == 2] = 1
                Y[Y == 5] = 2
                Y[Y == 13] = 3
                right = right+(c.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()

            print('test', epoch, right/su)

feature=pretrain()
finetune(feature)