import os
from pickle import NONE
import ssl

import torch
import torchvision
from torch import cuda, nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from lib.utils.util import get_logger
from lib.utils.data_split import createDIYDataset
from lib.utils.dataset import FabricDataset

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda:0' if cuda.is_available() else 'cpu')

'''
将print()替换为logger.info()，自动保存数据到./log/yyyy-mm-dd.log文件里
模型保存详情看代码注释
'''

# 获取日志工具
logger = get_logger()


class fabric(nn.Module):
    def __init__(self, num_classes=15):
        super(fabric, self).__init__()
        x = torchvision.models.AlexNet(num_classes)
        self.feature = x.features
        self.pool = x.avgpool
        self.classifier = x.classifier

    def forward(self, xx, xt):
        y1 = self.feature(xx)
        ytemp1 = self.feature(xt)
        y1 = self.pool(y1)
        ytemp1 = self.pool(ytemp1)
        pred = self.classifier(torch.flatten(y1-ytemp1, 1))
        return pred


def pretrain(trainRoot='./data/X_fabric_data/tr/', testRoot='./data/X_fabric_data/te/', classes=[1]):
    """
    @description  :预训练
    ---------
    @param  :
    trainRoot：训练集目录
    testRoot：测试集目录
    classes：预训练数据集包含的类别
    -------
    @Returns  :
    -------
    """
    transform = transforms.Grayscale(num_output_channels=3)
    train = FabricDataset(root=trainRoot, transform=transform)
    test = FabricDataset(root=testRoot, transform=transform)
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4,)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4,)

    MAX_EPOCHES = 30

    model = fabric(len(classes))

    model = model.to(device)
    if cuda.is_available():
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # 尽量加上这条代码，方便分辨数据记录
    logger.info('pretrain start training!')
    for epoch in range(MAX_EPOCHES):
        batch_idx = 0
        loss_sum = 0.0
        for X_gt, X_temp, Y in train:

            X_gt = X_gt.to(device)
            X_temp = X_temp.to(device)
            pred = model(X_gt, X_temp)

            for idx, c in enumerate(classes):
                Y[Y == c] = idx
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
                for idx, c in enumerate(classes):
                    Y[Y == c] = idx
                right += (pred.argmax(dim=1) == Y.to(device)
                          ).float().sum().cpu().item()
            logger.info(
                'Epoch:[{}/{}] batch_idx:{} loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, batch_idx, loss_sum/batch_idx, right/num))

    if cuda.is_available():
        return model.module.feature
    else:
        return model.feature


def finetune(feature,
             trainRoot='./data/DIY_fabric_data/tr/',
             testRoot='./data/DIY_fabric_data/te/',
             retrain=False, classes=[1, 2, 14],
             num_pretrain_classes=-1,
             feature_optim=False):
    """
    @description  :调优
    ---------
    @param  :
    feature：预训练得到的特征提取器
    trainRoot：训练集目录
    testRoot：测试集目录
    retrain：是否重新训练。True则重新训练，false则读取保存的模型继续训练
    classes：调优数据集包含的类别
    num_pretrain_classes：预训练数据集包含的类别个数
    -------
    @Returns  :
    -------
    """
    model_root = os.path.join('./models', 'model.pt')
    maxAccuracy = 0.0
    transform = transforms.Grayscale(num_output_channels=3)
    train = FabricDataset(root=trainRoot, transform=transform)
    test = FabricDataset(root=testRoot, transform=transform)
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 30

    classifier = torchvision.models.AlexNet(len(classes)).classifier
    avgpool = torchvision.models.AlexNet(len(classes)).avgpool

    if not retrain:
        if num_pretrain_classes == -1:
            logger.warn("please set num_pretrain_classes")
            return
        feature = torchvision.models.AlexNet(num_pretrain_classes).features

    if cuda.is_available():
        feature = nn.DataParallel(feature)
        avgpool = nn.DataParallel(avgpool)
        classifier = nn.DataParallel(classifier)

    # 读取已保存的模型
    if os.path.exists(model_root):
        checkpoint = torch.load(model_root)
        maxAccuracy = checkpoint['maxAccuracy']
        logger.info('Until now, max accuracy:{:.5f}'.format(maxAccuracy))
        if not retrain:
            feature.load_state_dict(checkpoint['feature'])
            classifier.load_state_dict(checkpoint['classifier'])
        else:
            if feature is None:
                logger.warn(
                    "please transfer feature which was got by pretrain into feature variable!")
                return

    feature = feature.to(device)
    avgpool = avgpool.to(device)
    classifier = classifier.to(device)

    optimizer = optim.Adam(classifier.parameters())
    feature_optimizer = optim.Adam(feature.parameters(), lr=0.0001)
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

            for idx, c in enumerate(classes):
                Y[Y == c] = idx
            Y = Y.to(device)

            loss = criterion(pred, Y)

            optimizer.zero_grad()
            if feature_optim == True:
                feature_optimizer.zero_grad()
            loss.backward()
            loss_sum += loss
            optimizer.step()
            if feature_optim == True:
                feature_optimizer.step()
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

                for idx, c in enumerate(classes):
                    Y[Y == c] = idx
                right = right+(pred.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()

            acc = right/sum
            logger.info(
                'Epoch:[{}/{}] batch_idx:{} loss={:.5f} accuracy={:.5f}'.format(epoch+1, MAX_EPOCHES, batch_idx, loss_sum/batch_idx, acc))

            # 模型保存，
            if acc > maxAccuracy:
                maxAccuracy = acc
                logger.info('acc:{:.5f} save model......'.format(acc))
                state = {'feature': feature.state_dict(
                ), 'classifier': classifier.state_dict(), 'maxAccuracy': acc}
                torch.save(state, os.path.join('./models', 'model.pt'))


if __name__ == "__main__":
    data_root = 'data/fabric_data_new'  # 原数据目录
    pretrain_data_root = 'data/X_data_new'  # 预训练数据根目录
    finetune_data_root = 'data/DIY_data_new'  # 调优数据根目录
    pretrain_classes = [1, 2, 3, 4, 5, 6, 7, 11,
                        14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]  # 预训练数据包含类别标签
    finetune_classes = [1, 2, 14]  # 调优数据包含类别标签
    ratio = 4.0/7  # 训练集所占比例

    # # 生成预训练数据集
    # createDIYDataset(root=os.path.join('.', data_root),
    #                  save=os.path.join('.',     pretrain_data_root),
    #                  classes=pretrain_classes,
    #                  ratio=ratio)

    # # 生成调优数据集
    # createDIYDataset(root=os.path.join('.', data_root),
    #                  save=os.path.join('.', finetune_data_root),
    #                  classes=finetune_classes,
    #                  ratio=ratio)

    retrain = False
    feature_optim = True
    if retrain:
        # 预训练
        feature = pretrain(trainRoot=os.path.join(pretrain_data_root, 'train'),
                           testRoot=os.path.join(pretrain_data_root, 'test'),
                           classes=pretrain_classes)
    else:
        feature = None

    # 调优
    finetune(feature=feature,
             trainRoot=os.path.join(finetune_data_root, 'train'),
             testRoot=os.path.join(finetune_data_root, 'test'),
             retrain=retrain,
             classes=finetune_classes,
             num_pretrain_classes=len(pretrain_classes),
             feature_optim=feature_optim)
