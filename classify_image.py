# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:12:01 2016

@author: osboxes
"""

from keras.models import model_from_json
import numpy as np

import matplotlib.pyplot as plt

model = model_from_json(open('./models/mnist_model_conv.json').read())
model.load_weights('./models/mnist_weights_conv.h5')

def inputChangeToConv(T):
    return T.reshape(T.shape[0], 1, 28, 28)

def printClass(data):
    plt.imshow(data.reshape((28,28)))
    pred = model.predict_classes(inputChangeToConv(data), batch_size=1)
    print 'Class is', pred

if __name__ == "__main__":
    data = plt.imread('number4_2.png').reshape((1,784))
    printClass(data)




