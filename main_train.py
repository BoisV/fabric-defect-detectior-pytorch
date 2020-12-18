import os
import ssl

import torch
import torchvision
from torch import cuda, nn, optim
from torch.utils.data.dataloader import DataLoader
from lib.utils.util import get_logger
from lib.utils.data_split import createDIYDataset
from lib.utils.dataset import FabricDataset

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

logger = get_logger()


class fabric(nn.Module):
    def __init__(self, num_classes=15):
        super(fabric, self).__init__()
        x = torchvision.models.AlexNet(num_classes)
        # self.feature = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        self.feature = x.features
        self.pool = x.avgpool
        # self.avgpool = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d((6, 6))
        # )
        self.classifier = x.classifier

    def forward(self, xx, xt):
        y1 = self.feature(xx)
        ytemp1 = self.feature(xt)
        y1 = self.pool(y1)
        ytemp1 = self.pool(ytemp1)
        pred = self.classifier(torch.flatten(y1-ytemp1, 1))
        return pred


def pretrain(trainRoot='./data/X_fabric_data/tr/', testRoot='./data/X_fabric_data/te/'):

    train = FabricDataset(root=trainRoot)
    test = FabricDataset(root=testRoot)
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 15

    model = fabric(15)

    model = model.to(device)
    if cuda.is_available():
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    logger.info('pretrain start training!')
    for epoch in range(MAX_EPOCHES):
        batch_idx = 0
        loss_sum = 0.0
        for X_gt, X_temp, Y in train:

            X_gt = X_gt.to(device)
            X_temp = X_temp.to(device)
            pred = model(X_gt, X_temp)

            Y = Y.to(device)
            loss = criterion(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss
            optimizer.step()
            batch_idx += 1

        with torch.no_grad():
            right = 0
            num = 0
            for X_gt, X_temp, Y in test:
                X_gt = X_gt.to(device)
                X_temp = X_temp.to(device)
                pred = model(X_gt, X_temp)
                num = num+Y.shape[0]

                right += (pred.argmax(dim=1) == Y.to(device)
                          ).float().sum().cpu().item()
            logger.info(
                'Epoch:[{}/{}] batch_idx:{} loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, batch_idx, loss_sum/batch_idx, right/num))

    if cuda.is_available():
        return model.module.feature
    else:
        return model.feature


def finetune(feature, trainRoot='./data/DIY_fabric_data/tr/', testRoot='./data/DIY_fabric_data/te/'):
    model_root = os.path.join('./models', 'model.pt')
    maxAccuracy = 0.0

    train = FabricDataset(root=trainRoot)
    test = FabricDataset(root=testRoot)
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 30

    classifier = torchvision.models.AlexNet(3).classifier
    avgpool = torchvision.models.AlexNet(3).avgpool

    if os.path.exists(model_root) and feature == None:
        checkpoit = torch.load(model_root)
        maxAccuracy = checkpoit['maxAccuracy']
        feature.load_state_dict(checkpoit['feature'])
        classifier.load_state_dict(checkpoit['classifer'])

    feature = feature.to(device)
    avgpool = avgpool.to(device)
    classifier = classifier.to(device)

    if cuda.is_available():
        feature = nn.DataParallel(feature)
        avgpool = nn.DataParallel(avgpool)
        classifier = nn.DataParallel(classifier)

    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    logger.info("finetune start training!")
    for epoch in range(MAX_EPOCHES):
        batch_idx = 0
        loss_sum = 0.0
        for X_gt, X_temp, Y in train:

            X_gt = X_gt.to(device)
            X_temp = X_temp.to(device)

            feature_gt = feature(X_gt)
            feature_temp = feature(X_temp)
            feature_gt = avgpool(feature_gt)
            feature_temp = avgpool(feature_temp)
            pred = classifier(torch.flatten(feature_gt-feature_temp, 1))

            Y[Y == 1] = 0
            Y[Y == 2] = 1
            Y[Y == 14] = 2
            Y = Y.to(device)

            loss = criterion(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            loss_sum += loss
            optimizer.step()
            # logger.info(
            #     'Epoch:[{}/{}] batch_idx:{} loss={:.5f}'.format(epoch, MAX_EPOCHES, batch_idx, loss))
            batch_idx += 1

        with torch.no_grad():

            right = 0
            sum = 0
            for X_gt, X_temp, Y in test:
                X_gt = X_gt.to(device)
                X_temp = X_temp.to(device)

                feature_gt = feature(X_gt)
                feature_temp = feature(X_temp)
                feature_gt = avgpool(feature_gt)
                feature_temp = avgpool(feature_temp)
                pred = classifier(torch.flatten(feature_gt-feature_temp, 1))

                sum = sum+Y.shape[0]

                Y[Y == 1] = 0
                Y[Y == 2] = 1
                Y[Y == 14] = 2
                right = right+(pred.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()

            acc = right/sum
            logger.info(
                'Epoch:[{}/{}] batch_idx:{} loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, batch_idx, loss_sum/batch_idx, acc))

            if acc > maxAccuracy:
                maxAccuracy = acc
                logger.info('acc:{:.5f} save model......'.format(acc))
                state = {'feature': feature.state_dict(
                ), 'classifier': classifier.state_dict(), 'maxAccuracy': acc}
                torch.save(state, os.path.join('./models', 'model.pt'))


if __name__ == "__main__":
    data_root = 'data/fabric_data'
    X_data_root = 'data/X_data'
    DIY_data_root = 'data/DIY_data'
    X_classes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ]
    three_classes = [1, 2, 14]

    # createDIYDataset(root=os.path.join('.', data_root),
    #                  save=os.path.join('.',     X_data_root),
    #                  classes=X_classes)
    # createDIYDataset(root=os.path.join('.', data_root),
    #                  save=os.path.join('.', DIY_data_root),
    #                  classes=three_classes)
    feature = pretrain(trainRoot=os.path.join(X_data_root, 'train'),
                       testRoot=os.path.join(X_data_root, 'test'))
    finetune(feature,
             trainRoot=os.path.join(DIY_data_root, 'train'),
             testRoot=os.path.join(DIY_data_root, 'test'))
