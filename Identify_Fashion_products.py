# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:25:20 2020

@author: ELCOT
"""

import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


filepath = 'train/' + str(1) + '.png'
img = imread(filepath)
img = img.astype('float32')
plt.imshow(img)


train_img = []
for img_name in train.id:
    img_path = 'train/' + str(img_name) + '.png'
    img = imread(img_path, as_gray=True)
    #as_gray remove channel column 
    img = img.astype('float32')
    train_img.append(img)

train_x = np.array(train_img)
train_x.shape

train_x = train_x/train_x.max()
train_x = train_x.reshape(-1, 28*28).astype('float32')
train_x.shape
train_y = train.label.values


import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.optim import Adam

input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# set remaining variables
epochs = 20
learning_rate = 0.0005


model = Sequential(Linear(input_num_units, hidden_num_units),
                   ReLU(),
                   Linear(hidden_num_units, output_num_units))
# loss function
loss_fn = CrossEntropyLoss()

# define optimization algorithm
optimizer = Adam(model.parameters(), lr=learning_rate)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, stratify = train_y)

train_losses = []
val_losses = []
for epoch in range(epochs):
    avg_cost = 0
    
    x, y = Variable(torch.from_numpy(train_x)), Variable(torch.from_numpy(train_y), requires_grad=False)
    x_val, y_val = Variable(torch.from_numpy(val_x)), Variable(torch.from_numpy(val_y), requires_grad=False)
    pred = model(x)
    pred_val = model(x_val)

    # get loss
    loss = loss_fn(pred, y)
    loss_val = loss_fn(pred_val, y_val)
    train_losses.append(loss)
    val_losses.append(loss_val)

    # perform backpropagation
    loss.backward()
    optimizer.step()
    avg_cost = avg_cost + loss.data

 

x, y = Variable(torch.from_numpy(train_x)), Variable(torch.from_numpy(train_y), requires_grad=False)
pred = model(x)

final_pred = np.argmax(pred.data.numpy(), axis=1)
accuracy_score(train_y, final_pred)


#Testing on single Image

img1_path = 'test/' + str(60001) + '.png'
img1 = imread(img1_path, as_gray = True)
plt.imshow(img1)

img1 = img1.astype('float32')
test_data = np.array(img1)
test_data = test_data.reshape(-1, 28*28).astype('float32')


test_pred = model(test_x)

test_pred = np,argmax(test_pred, axis = 1) # find maximum probality
