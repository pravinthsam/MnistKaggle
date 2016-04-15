# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 01:35:43 2016

@author: Pravinth Samuel
"""

import theano
import theano.tensor as T

import numpy as np

# Load the dataset
print 'Loading data from train.csv...'
import csv
with open('data/train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
    data = data[1:]
    
train_data_array = np.asarray(data, dtype = 'float64')
data = None

numRows = train_data_array.shape[0]
ratioTrain = 1.0
splitIx = int(ratioTrain*numRows)

train_set = (train_data_array[:splitIx,1:], np.asarray(train_data_array[:splitIx,0], dtype='int8'))
valid_set = (train_data_array[splitIx:,1:], np.asarray(train_data_array[splitIx:,0], dtype='int8'))

print 'Loading data from test.csv...'
with open('data/test.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
    data = data[1:]
    
test_set = np.asarray(data, dtype = 'float64')
data=None


X, Y = train_set

# ONE HOT ENCODING
Y_temp = np.zeros((Y.shape[0],10))
for i in range(Y.shape[0]):
    Y_temp[i,Y[i]] = 1
Y, Y_temp = Y_temp, None

# Normalizing X and remembering mu and sigma values
mu = None
sig = None

def normalize(mat):
    global mu, sig
    if mu is None:
        mu = np.mean(mat)
    if sig is None:
        sig = np.std(mat)
        
    return (mat - mu)/sig

X = normalize(X)

## Keras model

from keras.models import Sequential
from keras.layers.core import Dense, Activation

def numCorrect2(model, X,Y):
    pred = model.predict_classes(normalize(X), batch_size=10000)
    return np.sum(pred==Y)

def numCorrect3(model, X,Y):
    pred = model.predict_classes(X, batch_size=10000)
    return np.sum(pred==Y)


    
def printResults(model, numCorrect):
    correctTrain = numCorrect(model, train_set[0], train_set[1])
    correctVal = numCorrect(model, valid_set[0], valid_set[1])
    
    print("Training Correct: " + str(correctTrain) +
            " Incorrect: " + str(train_set[0].shape[0]-correctTrain))
    print("Validation Correct: " + str(correctVal) +
            " Incorrect: " + str(valid_set[0].shape[0]-correctVal))
            
from keras.callbacks import Callback
class ResultsPrinter(Callback):
    def __init__(self, m = None):
        if m is None:
            self.correctModel = numCorrect2
        else:
            self.correctModel = m
        
    def on_epoch_end(self, epoch, logs={}):
        printResults(self.model, self.correctModel)
        
        
mnistModel1 = Sequential()
mnistModel1.add(Dense(392, input_dim=784, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(196, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(49, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(10, init="glorot_uniform"))
mnistModel1.add(Activation("softmax"))

mnistModel1.compile(loss="categorical_crossentropy", optimizer="rmsprop")

rp = ResultsPrinter()        
mnistModel1.fit(X, Y, nb_epoch=15, batch_size=100, show_accuracy=True)

def saveModel(model, name):
    open('./models/' + name + '.json', 'w').write(model.to_json())
    model.save_weights('./models/' + name + '_weights.h5', overwrite=True)

saveModel(mnistModel1, 'mnist_simple_DNN_2')

raise Exception('this is the end, my beautiful friend')
## ConvNet

from keras.layers import Convolution2D, MaxPooling2D, Flatten

def inputChangeToConv(T):
    return T.reshape(T.shape[0], 1, 28, 28)
X = inputChangeToConv(train_set[0])
#X = train_set[0].reshape(train_set[0].shape[0], 1, 28, 28)

mnistModel2 = Sequential()
mnistModel2.add(Convolution2D(10, 3, 3, border_mode='valid', input_shape=(1, 28,28), init = 'glorot_uniform'))
mnistModel2.add(Activation("relu"))
mnistModel2.add(MaxPooling2D(pool_size=(2,2)))
mnistModel2.add(Convolution2D(10,3,3, init = 'glorot_uniform'))
mnistModel2.add(Activation('relu'))
mnistModel2.add(MaxPooling2D(pool_size=(2,2)))
mnistModel2.add(Convolution2D(10,3,3, init = 'glorot_uniform'))
mnistModel2.add(Activation('relu'))
mnistModel2.add(MaxPooling2D(pool_size=(2,2)))
mnistModel2.add(Flatten())
mnistModel2.add(Dense(10, init = 'glorot_uniform'))
mnistModel2.add(Activation("softmax"))

mnistModel2.compile(loss="categorical_crossentropy", optimizer="sgd")
mnistModel2.fit(X, Y, nb_epoch=100, batch_size=100, show_accuracy=True)


def numCorrect4(model, X,Y):
    pred = model.predict_classes(inputChangeToConv(X), batch_size=10000)
    return np.sum(pred==Y)

printResults(mnistModel2, numCorrect4)

open('./models/mnist_model_conv.json', 'w').write(mnistModel2.to_json())
mnistModel2.save_weights('./models/mnist_weights_conv.h5')


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
