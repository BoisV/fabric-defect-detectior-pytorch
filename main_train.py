from torch import nn
from torch.utils.data.dataloader import DataLoader
from lib.models.model import vgg16
from lib.utils.dataset import FabricDataset
import torch
from torchvision import transforms
from torch import cuda, optim
if __name__ == "__main__":
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    model = vgg16(num_classes=4, init_weights=False)
    model = model.to(device=device)

    transform = transforms.Compose([transforms.Resize(size=[400,400]),
                                    transforms.ToTensor()])
    dataset = FabricDataset(root='./data/DIY_fabric_data',
                            train=True, transform=transform)
    train_iter = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    
    MAX_EPOCHES = 100
    lr = 0.1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(MAX_EPOCHES):
        for X, Y, temp, coors in train_iter:
            X = X.to(device)
            pred = model(X)
            pred = pred.to('cpu')
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            print('%.6f', loss)
