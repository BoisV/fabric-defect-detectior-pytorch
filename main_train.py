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
        self.pretrained = nn.Sequential(
            torchvision.models.vgg11(pretrained=True).features,
            nn.AdaptiveAvgPool2d((7, 7)))

        self.fc = nn.Linear(512 * 7 * 7, 4)

    def forward(self, xx, xt):
        y = self.pretrained(xx)
        ytemp = self.pretrained(xt)
        c = self.fc(torch.flatten(y-ytemp, 1))
        return c


if __name__ == "__main__":
    # tainsform未生效
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train = FabricDataset(root='./data/DIY_fabric_data/train/')
    test = FabricDataset(root='./data/DIY_fabric_data/test/')
    train = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    test = DataLoader(test, batch_size=128, shuffle=True, num_workers=4)

    MAX_EPOCHES = 100
    lr = 0.1

    model = fabric()
    model = model.to(device)
    if cuda.is_available():
        model = torch.nn.DataParallel(model)
    '''
    opt1=torch.optim.Adam([
            {'params':model.module.pretrained.parameters(),'lr':0.000001},
            
            {'params':model.module.fc.parameters(),'lr':0.01}
            ])
    '''
    opt1 = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(MAX_EPOCHES):
        for xx, xt, Y in train:

            xx = xx.to(device)
            xt = xt.to(device)
            c = model(xx, xt)

            Y[Y == 1] = 0
            Y[Y == 2] = 1
            Y[Y == 5] = 2
            Y[Y == 13] = 3
            Y = Y.to(device)
            '''
            _onehot=torch.zeros((Y.shape[0],14),dtype=torch.int64)
                
            _onehot=_onehot.scatter_(1,_onehot,1)
            _onehot=_onehot[:,[1,2,5,13]]
            _onehot=_onehot.to(device)
            '''
            loss = criterion(c, Y)

            opt1.zero_grad()
            loss.backward()

            opt1.step()
            print(loss.item())

        with torch.no_grad():

            right = 0
            su = 0
            for X, Y, temp, xx, xt in test:
                X = X.to(device)
                temp = temp.to(device)
                xx = xx.to(device)
                xt = xt.to(device)
                c = model(X, temp, xx, xt)
                su = su+Y.shape[0]

                Y[Y == 1] = 0
                Y[Y == 2] = 1
                Y[Y == 5] = 2
                Y[Y == 13] = 3

                right = right+(c.argmax(dim=1) == Y.to(device)
                               ).float().sum().cpu().item()

            print('test', epoch, right/su)
