# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:00:30 2016

@author: pravinth
"""
#### Simple logistic regression model

import theano
import theano.tensor as T

import numpy as np

# Load the dataset
print 'Reading data from train.csv...'
import csv
with open('data/train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
    data = data[1:]
    
train_data_array = np.asarray(data, dtype = 'float64')
data = None

train_set = (train_data_array[:30000,1:], np.asarray(train_data_array[:30000,0], dtype='int8'))
test_set = (train_data_array[30000:,1:], np.asarray(train_data_array[30000:,0], dtype='int8'))



with open('data/test.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',', 
                           quotechar = '"')
    data = [data for data in data_iter]
    data = data[1:]
    
ulti_test_set = np.asarray(data, dtype = 'float64')
data=None


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

n_iter = 1001



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
