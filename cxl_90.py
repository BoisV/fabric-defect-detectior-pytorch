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
        x=torchvision.models.AlexNet(15)
        self.feature1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0), #change kernel_size form 11 to 7, stride from 4 to 2, padding from 2 to 0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),   #add a MaxPooling
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            
        )
        self.avgpool=nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),    #change stride from 2 to 1
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier=x.classifier

    def forward(self, xx, xt):
        y1 = self.feature1(xx)
        ytemp1 = self.feature1(xt)
        y1=self.avgpool(y1)
        ytemp1=self.avgpool(ytemp1)
        c = self.classifier(torch.flatten(y1-ytemp1, 1))
        return c


def pretrain():
    
    train = FabricDataset(root='./data/X_fabric_data/tr/')
    test = FabricDataset(root='./data/X_fabric_data/te/')
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 30
 
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
            print("loss = ",loss.item())

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

            print('pretrain test: epoch=', epoch,"accurate:", right/su)
    if cuda.is_available():
        return model.module.feature1
    else:
        return model.feature1

def finetune(feature):
    classifier=torchvision.models.AlexNet(4).classifier

    train = FabricDataset(root='./data/DIY_fabric_data/tr/')
    test = FabricDataset(root='./data/DIY_fabric_data/te/')
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 60
  
    
    feature = feature.to(device)

    classifier=classifier.to(device)
    if cuda.is_available():
        feature = torch.nn.DataParallel(feature)

        classifier = torch.nn.DataParallel(classifier)
    
    opt1=torch.optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    avgpool=torchvision.models.AlexNet(4).avgpool
    
    max_accu = 0
    max_epoch = 0
    for epoch in range(MAX_EPOCHES):
        for xx, xt, Y in train:

            xx = xx.to(device)
            xt = xt.to(device)
            
            

            y1 = feature(xx)
            ytemp1 = feature(xt)
            y1=avgpool(y1)
            ytemp1=avgpool(ytemp1)
            c = classifier(torch.flatten(y1-ytemp1, 1))



            Y[Y == 1] = 0
            Y[Y == 2] = 1
            Y[Y == 5] = 2
            Y[Y == 13] = 3
            Y = Y.to(device) 
            loss = criterion(c, Y)

            opt1.zero_grad()
            loss.backward()

            opt1.step()
            print("loss = ",loss.item())

        with torch.no_grad():

            right = 0
            su = 0
            for xx, xt, Y in test: 
                xx = xx.to(device)
                xt = xt.to(device)
                
                y1 = feature(xx)
                ytemp1 = feature(xt)
                y1=avgpool(y1)
                ytemp1=avgpool(ytemp1)
                c = classifier(torch.flatten(y1-ytemp1, 1))


                su = su+Y.shape[0]

                Y[Y == 1] = 0
                Y[Y == 2] = 1
                Y[Y == 5] = 2
                Y[Y == 13] = 3
                right = right+(c.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()
            accu = right/su
            print('finetune test: epoch=', epoch,"accurate:", accu)
            if(accu > max_accu):
                max_accu = accu
                max_epoch = epoch
    print("max_accurate=",max_accu,"in epoch",max_epoch)
    
feature=pretrain()
finetune(feature)