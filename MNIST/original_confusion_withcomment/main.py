import pandas as pd
import matplotlib.pyplot as plt
import torch
# nn means neural network: the module of building neural network in torch
import torch.nn as nn
import torch.optim as optim
# 和torch.nn类似 不过nn.functional.xxx是函数的接口 torch.nn是nn.functional的类封装 它俩共同祖先nn.Module
# F is a function in nn, if do not write the sentence below, if we need use: nn.functional.xxx, and now: F.xxx. Be simplify
import torch.nn.functional as F
# is a individual tool library from PyTorch which can support some module to do image manipulation
import torchvision
# a packsge which include some normal way to transform the images
from torchvision import transforms
# a tool which can process the input data
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from getdata import read_data, split_data



# Custom class for the MNIST dataset from Kaggle
# Images come in a csv, not as actual images
# Training set is split before given to this class
class MNISTDataSet(torch.utils.data.Dataset):
    # image df, labels df, transforms
    # uses labels to determine if it needs to return X & y or just X in __getitem__
    def __init__(self, images, labels, transforms = None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # gets the row
        data = self.X.iloc[i, :]
        # reshape the row into the image size
        #(numpy array have the color channels dim last)
        data = np.array(data).astype(np.uint8).reshape(28, 28, 1)

        # perform transforms if they are any
        if self.transforms:
            data = self.transforms(data)

        # if !test_set return the label as well, otherwise don't
        if self.y is not None:
            return (data,self.y[i]) #train/val
        else:
            return data 


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # image starts as (1, 28, 28)
        # Formula to computer size of image after conv/pool
        # (sixe-filter + 2 * padding / stride) + 1
        # stride: 返回tensor的步长 padding: 定义一个常数来对图像或者张量的边缘进行填充，若该常数等于0则等价于0填充
        # 卷积完的尺寸取决于原图像的尺寸 和卷积核无关
                                # inputs          # of filters   filter size
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 32, kernel_size=5, stride=1, padding=2) # conv1
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size=5, stride=1, padding=2) # conv2
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size=5, stride=1, padding=2) # conv3
        self.conv3_bn = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(in_features=128*6*6, out_features=1024) # linear 1
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512) # linear 2
        self.fc2_bn = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256) # linear 3
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128) # linear 4
        self.fc4_bn = nn.BatchNorm1d(num_features=128)
        self.out = nn.Linear(in_features=128, out_features=10) # output layer

    def forward(self, t):
        # 这里的relu是active function
        # max_pool2d (最大)池化 kernel_size(int or tuple):max pooling窗口大小 stride:maxpooling移动步长
        # max_pool2d作用：1.特征不变 2.特征降维 3，防止过拟合
        t = F.relu(self.conv1_bn(self.conv1(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2) # (1, 14, 14)

        t = F.relu(self.conv2_bn(self.conv2(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2) # (1, 7, 7)

        t = F.relu(self.conv3_bn(self.conv3(t)))
        t = F.max_pool2d(t, kernel_size = 2, stride = 1) # (1, 6, 6)

        # print(t.shape)

        # t = t.view(-1, 128*6*6)
        # print(t.shape)
        # t = t.reshape(-1, 128*6*6)
        # print(t.shape)
        # t = F.relu(self.fc1_bn(self.fc1(t)))

        t = F.relu(self.fc1_bn(self.fc1(t.reshape(-1, 128*6*6))))
        t = F.relu(self.fc2_bn(self.fc2(t)))
        t = F.relu(self.fc3_bn(self.fc3(t)))
        t = F.relu(self.fc4_bn(self.fc4(t)))
        t = self.out(t)

        return t



def main():
    train_set, test_images = read_data()
    train_images, val_images, train_labels, val_labels = split_data()

    # get the GPU if there is one, otherwise the cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)

    #Quick function that gets how many out of "preds" match "label"
    def get_num_correct(preds, labels):
        return preds.argmax(dim = 1).eq(labels).sum().item()

    # Set up
    # Transformations: size of images in MNIST
    IMG_SIZE = 28
    # Also the images only have one color channel, so 3D size = (1, 28, 28)

    
    # Transformations for the train. transforms.ToTensor(): divides by 255. transforms.Normalize((0.5,), (0.5,))
    # transforms function 是将多个步骤整合在一起，任何图像增强或者归一化都需要用到此模块
    # ToPILImage: 将tensor转换为PIL Image； RandomCrop: 在图片的中间区域进行裁剪； ToTensor: convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
    train_trans = transforms.Compose(([transforms.ToPILImage(), transforms.RandomCrop(IMG_SIZE), transforms.ToTensor()]))
    # Transformations for the validation & test sets. transforms.ToTensor(): divides by 255. transforms.Normalize((0.1307,), (0.3081,))
    val_trans = transforms.Compose(([transforms.ToPILImage(), transforms.ToTensor()]))


    # Get datasets using the custom(自定义) MNIST Dataset for the train, val, and test images
    train_set = MNISTDataSet(train_images, train_labels, train_trans)
    val_set = MNISTDataSet(val_images, val_labels, val_trans)
    test_set = MNISTDataSet(test_images, None, val_trans)
    


    # Training Loop
    lr = 0.001 # initial learning rate
    batch_size = 100
    epochs = 10 # number of epochs to run

    """network = Network().to(device) # put the model on device (hopefully a GPU! But I don't have)
    # shuffle default False 若True，在每个epoch中对数据集data进行重排
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in range(epochs):
        print('Epoch:', epoch+1)
        epoch_loss = 0
        epoch_correct = 0
        network.train() # train mode

        # lessen the learning rate after 4 epochs (0, 1, 2, 3)
        if epoch == 4:
            print('decreasing lr')
            optimizer = optim.Adam(network.parameters(), lr = 0.00001)

        if epoch == 10: # not currently used
            print('decreasing lr')
            optimizer = optim.Adam(network.parameters(), lr = 0.0000000000001)

        for images, labels in tqdm(train_dl):
            X, y = images.to(device), labels.to(device) # put X & y on device
            y_ = network(X) # get predictions

            optimizer.zero_grad() # zeros out the gradients
            loss = F.cross_entropy(y_, y) # computes the loss
            loss.backward() # computes the gradients
            optimizer.step() # updates weights

            epoch_loss += loss.item() * batch_size
            epoch_correct += get_num_correct(y_, y)

        # Evaluation with the validation set
        # 训练模型使用network.train() 测试模型使用network.eval()
        network.eval() # eval mode
        val_loss = 0
        val_correct = 0
        # with torch.no_grad()用于停止autograd模块的工作，以起到加速和节省显存的作用
        # 具体行为就是停止gradient的计算，从而节省GPU算力和显存，但是不会影响dropout和batchnorm层的行为
        with torch.no_grad():
            for images, labels in val_dl:
                X, y = images.to(device), labels.to(device) # to device

                preds = network(X) #get predictions
                loss = F.cross_entropy(preds, y) # calculate the loss

                val_correct += get_num_correct(preds, y)
                val_loss += loss.item() * batch_size
        # print the loss and accuracy for the validation set
        print('Val Loss: ', val_loss)
        print('Val Acc: ', (val_correct/len(val_images))*100)"""









    network = Network().to(device)
    network.load_state_dict(torch.load('./original_confusion_withcomment/trainingresult/saved.pt'))
    network.eval()


    # Time to get the network's predictions on the test set
    # Put the test set in a DataLoader
    test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    network.eval() # Safety first
    predictions = torch.LongTensor().to(device) # Tensor for all predictions

    # Go through the test set, saving the predictions in... 'predictions'
    for images in test_dl:
        preds = network(images.to(device))
        predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)



    # fig, ax = plt.subplots(nrows = 5, ncols = 5, figsize = (8, 8))
    # fig.subplots_adjust(hspace = .3)
    # for i in range(5):
    #     for j in range(5):
    #         ax[i][j].axis('off')
    #         ax[i][j].imshow(test_images.iloc[i+(j*5), :].to_numpy().astype(np.uint8).reshape(28, 28), cmap = 'gray')
    #         ax[i][j].set_title(predictions[i+(j*5)])
    # plt.show()


    for i in range(len(images)):
        plt.imshow(test_images.iloc[i, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap = 'gray')
        plt.title(predictions[i])
        plt.show()


if __name__ == '__main__':
    main()