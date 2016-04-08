# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 01:35:43 2016

@author: pravinth
"""

import theano
import theano.tensor as T

import cPickle, gzip
import numpy as np

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()



n = train_set[0].shape[0]
m = train_set[0].shape[1]

train_set = (train_set[0][:n],train_set[1][:n])

X, Y = train_set

Y_temp = np.zeros((n,10))
for i in range(n):
    Y_temp[i,Y[i]] = 1
Y, Y_temp = Y_temp, None

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

x = T.dmatrix("x")
y = T.dmatrix("y")

w = theano.shared(np.random.randn(m,10)*0.01, name="w")

b = theano.shared(np.zeros(10), name="b")

out_unnormalized = T.exp(T.dot(x,w)+b)

out_normalized = out_unnormalized / (T.sum(out_unnormalized, axis=1, keepdims=True))
likelihood = T.sum(out_normalized*y)

cost = -T.log(likelihood)

gradW, gradB = T.grad(cost, [w, b])

n_iter = 11



train = theano.function(inputs = [x, y],
                        outputs = cost,
                        updates = [(w, w - 0.01*gradW),(b,b-0.01*gradB)])

predict = theano.function(inputs = [x],
                          outputs = out_normalized)

def numCorrect1(X,Y):
    pred = predict(normalize(X))
    numCorrect = np.sum(np.argmax(pred,axis=1)==Y)
    return numCorrect

for itern in range(n_iter):
    c = train(X,Y)
    
    if itern%10 == 0:
        correctTrain = numCorrect1(train_set[0], train_set[1])
        correctTest = numCorrect1(test_set[0], test_set[1])        
        print 'cost: ' + str(c) + ' correctTrain: ' + str(correctTrain) + ' correctTest: ' + str(correctTest)
        
## Keras model

from keras.models import Sequential
from keras.layers.core import Dense, Activation

def numCorrect2(model, X,Y):
    pred = model.predict_classes(normalize(X), batch_size=100)
    return np.sum(pred==Y)

def numCorrect3(model, X,Y):
    pred = model.predict_classes(X, batch_size=100)
    return np.sum(pred==Y)


    
def printResults(model, numCorrect):
    correctTrain = numCorrect(model, train_set[0], train_set[1])
    correctVal = numCorrect(model, valid_set[0], valid_set[1])
    correctTest = numCorrect(model, test_set[0], test_set[1])
    
    print("Training Correct: " + str(correctTrain) +
            " Incorrect: " + str(train_set[0].shape[0]-correctTrain))
    print("Validation Correct: " + str(correctVal) +
            " Incorrect: " + str(valid_set[0].shape[0]-correctVal))
    print("Test Correct: " + str(correctTest) +
            " Incorrect: " + str(test_set[0].shape[0]-correctTest))
'''
mnistModel1 = Sequential()
mnistModel1.add(Dense(392, input_dim=784, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(196, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(49, init="glorot_uniform"))
mnistModel1.add(Activation("relu"))
mnistModel1.add(Dense(10, init="glorot_uniform"))
mnistModel1.add(Activation("softmax"))

mnistModel1.compile(loss="categorical_crossentropy", optimizer="sgd")
mnistModel1.fit(train_set[0], Y, nb_epoch=5, batch_size=10, show_accuracy=True)
'''

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


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        