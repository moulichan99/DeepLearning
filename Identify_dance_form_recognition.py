import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from zipfile import ZipFile
filename = 'Dance Form.zip'

with ZipFile(filename,'r') as zip:
   zip.extractall()
   print('done') 


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

from skimage.transform import resize
filepath = 'train/' + str(1) + '.jpg'
print(filepath)
img = imread(filepath)
img = resize(img, (128, 128))
img = img.astype('float32')
plt.imshow(img)

train_img = []
for img_name in train.Image:
    filepath = 'train/' + img_name
    img = imread(filepath)
    img = resize(img, (128, 128))
    #as_gray remove channel column 
    img = img.astype('float32')
    train_img.append(img)


train_x = np.array(train_img)
train_x.shape

train_x = train_x/255

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_y = lb.fit_transform(train.target) 

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn


learning_rate = 0.0005
train_x = train_x.reshape(364, 3, 128, 128)


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3,6,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6,16,5)
    self.fc1 = nn.Linear(16*29*29, 500)
    self.fc2 = nn.Linear(500, 120)
    self.fc3 = nn.Linear(120, 8)
  
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16*29*29)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(6),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(16 * 29 * 29, 8)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = ConvNet()
print(model)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

train_losses = []
epochs= 20
for epoch in range(epochs):
    avg_cost = 0
    
    x, y = Variable(torch.from_numpy(train_x)), Variable(torch.from_numpy(train_y))
    
    optimizer.zero_grad()
    pred = model(x)

    # get loss
    loss = loss_fn(pred, y)
    train_losses.append(loss)
    # perform backpropagation
    loss.backward()
    optimizer.step()
    avg_cost = avg_cost + loss.data

    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss)

 x, y = Variable(torch.from_numpy(train_x)), Variable(torch.from_numpy(train_y), requires_grad=False)
pred = model(x)

final_pred = np.argmax(pred.data.numpy(), axis=1)
accuracy_score(train_y, final_pred)
