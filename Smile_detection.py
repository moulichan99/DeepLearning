import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
'''
1 - Emotion Tracking
A nearby community health clinic is helping the local residents monitor their mental health.
As part of their study, they are asking volunteers to record their emotions throughout the day.
To help the participants more easily track their emotions, you are asked to create an app that will classify their emotions based on some pictures that the volunteers will take of their facial expressions.
As a proof-of-concept, you first train your model to detect if someone's emotion is classified as "happy" or "not happy." 
To build and train this model, you have gathered pictures of some volunteers in a nearby neighborhood. The dataset is labeled
'''
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


'''
2 - Building a model in Keras
Keras is very good for rapid prototyping. In just a short time you will be able to build a model that achieves outstanding results.

Here is an example of a model in Keras:

def model(input_shape):
    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model

    Variable naming convention
Note that Keras uses a different convention with variable names than we've previously used with numpy and TensorFlow.
Instead of creating unique variable names for each step and each layer, such as
X = ...
Z1 = ...
A1 = ...
Keras re-uses and overwrites the same variable at each step:
X = ...
X = ...
X = ...
The exception is X_input, which we kept separate since it's needed later.
Objects as functions
Notice how there are two pairs of parentheses in each statement. For example:
X = ZeroPadding2D((3, 3))(X_input)
The first is a constructor call which creates an object (ZeroPadding2D).
In Python, objects can be called as functions. Search for 'python object as function and you can read this blog post Python Pandemonium. See the section titled "Objects as functions."
The single line is equivalent to this:
ZP = ZeroPadding2D((3, 3)) # ZP is an object that can be called as a function
X = ZP(X_input)
Exercise: Implement a HappyModel().

This assignment is more open-ended than most.
Start by implementing a model using the architecture we suggest, and run through the rest of this assignment using that as your initial model. * Later, come back and try out other model architectures.
For example, you might take inspiration from the model above, but then vary the network architecture and hyperparameters however you wish.
You can also use other functions such as AveragePooling2D(), GlobalMaxPooling2D(), Dropout().

'''

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """


    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    X_input  = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    ### END CODE HERE ###
    
    return model

'''Step 1: create the model.
Hint:
The input_shape parameter is a tuple (height, width, channels). It excludes the batch number.
Try X_train.shape[1:] as the input_shape.
'''
### START CODE HERE ### (1 line)
happyModel = HappyModel(X_train.shape[1:])
### END CODE HERE ###
'''
Step 2: compile the model
Hint:
Optimizers you can try include 'adam', 'sgd' or others. See the documentation for optimizers
The "happiness detection" is a binary classification problem. The loss function that you can use is 'binary_cross_entropy'. Note that 'categorical_cross_entropy' won't work with your data set as its formatted, because the data is an array of 0 or 1 rather than two arrays (one for each category). Documentation for losses
'''
### START CODHERE ### (1 line)
happyModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])
### END CODE HERE ###

'''
Step 3: train the model
Hint:
Use the 'X_train', 'Y_train' variables. Use integers for the epochs and batch_size
Note: If you run fit() again, the model will continue to train with the parameters it has already learned instead of reinitializing them.
'''
### START CODE HERE ### (1 line)
happyModel.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 64)
### END CODE HERE ###
'''
Step 4: evaluate model
Hint:
Use the 'X_test' and 'Y_test' variables to evaluate the model's performance.
'''
### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
150/150 [==============================] - 1s     

Loss = 0.10602962623
Test Accuracy = 0.939999997616