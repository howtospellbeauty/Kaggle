import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from getdata import split_data
from torch.optim.lr_scheduler import MultiStepLR


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MNISTDataSet(Dataset):
    def __init__(self, images, labels, transforms = None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1)

        if self.transforms:
            data = self.transforms(data)
        if self.y is not None:
            return (data,self.y[i]) 
        else:
            return data 


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 32, kernel_size=5, stride=1, padding=2) 
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size=5, stride=1, padding=2) 
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size=5, stride=1, padding=2) 
        self.conv3_bn = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(in_features=128*6*6, out_features=1024) 
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512) 
        self.fc2_bn = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256) 
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128) 
        self.fc4_bn = nn.BatchNorm1d(num_features=128)
        self.out = nn.Linear(in_features=128, out_features=10) 

    def forward(self, t):
        t = F.relu(self.conv1_bn(self.conv1(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2) 

        t = F.relu(self.conv2_bn(self.conv2(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2) 

        t = F.relu(self.conv3_bn(self.conv3(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 1) 

        t = F.relu(self.fc1_bn(self.fc1(t.reshape(-1, 128*6*6))))
        t = F.relu(self.fc2_bn(self.fc2(t)))
        t = F.relu(self.fc3_bn(self.fc3(t)))
        t = F.relu(self.fc4_bn(self.fc4(t)))
        t = self.out(t)

        return t

def get_num_correct(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()
    
def train_epoch(network, train_dl, optimizer):
    epoch_loss = 0.0
    epoch_correct = 0
    for images, labels in tqdm(train_dl):
        X, y = images.to(device), labels.to(device) 
        y_ = network(X) 

        optimizer.zero_grad() 
        loss = F.cross_entropy(y_, y) 
        loss.backward() 
        optimizer.step() 

        epoch_loss += loss.item() * X.shape[0]
        epoch_correct += get_num_correct(y_, y)

def eval_epoch(network, val_dl, val_images):
    with torch.no_grad():
        network.eval()
        val_loss = 0
        val_correct = 0
        for images, labels in val_dl:
            X, y = images.to(device), labels.to(device) 

            preds = network(X) 
            loss = F.cross_entropy(preds, y) 

            val_correct += get_num_correct(preds, y)
            val_loss += loss.item() * X.shape[0]

    print('Val Loss: ', val_loss)
    print('Val Acc: ', (val_correct/len(val_images))*100)

    return val_loss



def main():
    train_data = pd.read_csv('./final_model/data/train.csv')
    test_images = pd.read_csv('./final_model/data/test.csv')
    train_images, val_images, train_labels, val_labels = split_data(train_data)
    transform = transforms.Compose(([transforms.ToPILImage(), transforms.ToTensor()]))
    train_set = MNISTDataSet(train_images, train_labels, transform)
    val_set = MNISTDataSet(val_images, val_labels, transform)
    test_set = MNISTDataSet(test_images, None, transform)
    train_dl = DataLoader(train_set, batch_size=100, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=100, shuffle=False)

    ######
    lr = 0.001 
    epochs = 10 

    network = Network().to(device)

    optimizer = optim.Adam(network.parameters(), lr = 0.001)
    scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)

    ######
    for epoch in range(epochs):
        network.train()

        train_epoch(network, train_dl, optimizer)
        scheduler.step()

        val_loss = eval_epoch(network, val_dl, val_images)
        # torch.save(network.state_dict(), f'/trainingresult/{epoch}-{val_loss}.pt')
        torch.save(network.state_dict(), f'./final_model/trainingresult/saved.pt')
    ######
    

def predict():
    network = Network().to(device)
    network.load_state_dict(torch.load('./final_model/trainingresult/saved.pt'))
    network.eval()
    test_images = pd.read_csv('./final_model/data/test.csv')
    transform = transforms.Compose(([transforms.ToPILImage(), transforms.ToTensor()]))
    test_set = MNISTDataSet(test_images, None, transform)

    test_dl = DataLoader(test_set, batch_size=100, shuffle=False)

    predictions = torch.LongTensor().to(device)

    for images in test_dl:
        preds = network(images.to(device))
        predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)

    for i in range(len(images)):
        plt.imshow(test_images.iloc[i, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap = 'gray')
        plt.title(predictions[i])
        plt.show()


if __name__ == '__main__':
    # main()
    predict()